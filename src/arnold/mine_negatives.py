#!/usr/bin/env python3
"""
Hard Negative Mining — оценка политики на всех motion и отбор провальных.

Два режима:
  1. mine   — eval + filter + save (default)
  2. refilter — пересчитать фильтр из существующего JSON без повторного eval

Результат:
  - {output_name}_{split}.txt — motion_id по одному на строку (для MyoLegsIm)
  - {output_name}_{split}.json — полная мета + per-motion метрики

Критерии негатива задаются раздельно для train и test:
  run.train.mean_mpjpe=0.05   (строже — модель видела эти motions)
  run.test.mean_mpjpe=0.08    (мягче — unseen data)

Использование:

    # Mine (eval + filter)
    poetry run python -m arnold.mine_negatives \
        run=mine_negatives \
        '+run/experts@run.experts.myo=myohuman' \
        resume_checkpoint=runs/exp/model.pth

    # Refilter (no eval, just re-apply thresholds to existing JSON)
    poetry run python -m arnold.mine_negatives refilter \
        --json data/trained_models/exp/negatives_valid.json \
        --mean-mpjpe 0.06 --not-success \
        --output data/trained_models/exp/negatives_valid_strict.txt
"""

import os
import sys
import json
import logging
import argparse

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Filter
# ─────────────────────────────────────────────────────────────────────

def filter_negatives(
    results: list,
    mean_mpjpe: float = None,
    max_mpjpe: float = None,
    not_success: bool = False,
) -> list:
    """
    Фильтрует per-motion results по критериям негатива (OR).
    """
    negatives = []
    for r in results:
        is_negative = False

        if not_success and not r.get("success", True):
            is_negative = True
        if mean_mpjpe is not None and r.get("mpjpe") is not None:
            if r["mpjpe"] > mean_mpjpe:
                is_negative = True
        if max_mpjpe is not None and r.get("max_mpjpe") is not None:
            if r["max_mpjpe"] > max_mpjpe:
                is_negative = True

        if is_negative:
            negatives.append(r)

    return negatives


# ─────────────────────────────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────────────────────────────

def save_txt(negatives: list, total: int, path: str) -> None:
    """Save plain txt with motion IDs."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    neg_ids = sorted(int(r["motion_id"]) for r in negatives)
    with open(path, "w") as f:
        f.write(f"# {len(neg_ids)} negatives / {total} total\n")
        for mid in neg_ids:
            f.write(f"{mid}\n")


def save_json(per_motion: list, negatives: list, meta: dict, path: str) -> None:
    """Save full JSON with meta + per-motion results."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    output = {"meta": meta, "negatives": negatives, "all_results": per_motion}
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)


def save_results(
    per_motion: list,
    negatives: list,
    output_dir: str,
    output_name: str,
    split: str,
    expert_name: str,
    checkpoint_path: str,
    epoch: int,
    criteria: dict,
) -> str:
    """Save txt + json. Returns txt path."""
    base = f"{output_name}_{split}"
    if expert_name:
        base = f"{base}_{expert_name}"
    txt_path = os.path.join(output_dir, f"{base}.txt")
    json_path = os.path.join(output_dir, f"{base}.json")

    save_txt(negatives, len(per_motion), txt_path)

    meta = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "epoch": epoch,
        "expert_name": expert_name,
        "total_motions": len(per_motion),
        "negatives_count": len(negatives),
        **criteria,
    }
    save_json(per_motion, negatives, meta, json_path)

    return txt_path


# ─────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────

def log_group_stats(items: list, label: str) -> None:
    if not items:
        logger.info(f"  {label}: (empty)")
        return
    mpjpes = [r["mpjpe"] for r in items if r.get("mpjpe") is not None]
    max_mpjpes = [r["max_mpjpe"] for r in items if r.get("max_mpjpe") is not None]
    coverages = [r["frame_coverage"] for r in items if r.get("frame_coverage") is not None]
    rewards = [r["reward"] for r in items if r.get("reward") is not None]
    successes = [r["success"] for r in items if r.get("success") is not None]

    parts = [f"{label} ({len(items)})"]
    if successes:
        parts.append(f"success={np.mean(successes):.1%}")
    if mpjpes:
        parts.append(f"mean_mpjpe={np.mean(mpjpes) * 1000:.1f}mm")
    if max_mpjpes:
        parts.append(f"max_mpjpe={np.mean(max_mpjpes) * 1000:.1f}mm")
    if coverages:
        parts.append(f"frame_cov={np.mean(coverages):.3f}")
    if rewards:
        parts.append(f"reward={np.mean(rewards):.1f}±{np.std(rewards):.1f}")
    logger.info(f"  {' | '.join(parts)}")


def log_summary(expert_name: str, split: str, per_motion: list, negatives: list) -> None:
    total = len(per_motion)
    n_neg = len(negatives)
    neg_ids = {id(r) for r in negatives}
    positives = [r for r in per_motion if id(r) not in neg_ids]

    logger.info(
        f"[{expert_name}] {split}: "
        f"{n_neg} negatives + {len(positives)} positives = {total} total"
    )
    log_group_stats(negatives, "negatives")
    log_group_stats(positives, "positives")


# ─────────────────────────────────────────────────────────────────────
#  Mine (eval + filter)
# ─────────────────────────────────────────────────────────────────────

def _evaluate_all_experts(trainer, split):
    """
    Eval all experts with return_per_motion=True.

    For split="train": temporarily swaps valid wrappers to train wrappers
    so vectorized workers use the training env.

    Returns:
        {expert_name: (aggregate_metrics, per_motion_results)}
    """
    results = {}

    saved_valid = None
    if split == "train":
        saved_valid = trainer.valid_experts
        trainer.valid_experts = {
            n: ctx.wrapper for n, ctx in trainer.experts.items()
        }
        for ctx in trainer.experts.values():
            ctx.wrapper.env.start_eval(im_eval=True)

    try:
        for expert_name, ctx in trainer.experts.items():
            wrapper = trainer.valid_experts.get(expert_name)
            if wrapper is None:
                raise ValueError(
                    f"No wrapper for expert '{expert_name}'. "
                    f"Set eval_frequency > 0 in config."
                )

            if trainer.vectorized_eval and trainer.device.type in ('cuda', 'mps'):
                metrics, per_motion = trainer._evaluate_expert_vectorized(
                    expert_name, ctx, wrapper, return_per_motion=True,
                )
            else:
                metrics, per_motion = trainer.evaluate_expert(
                    expert_name, ctx, wrapper, return_per_motion=True,
                )
            results[expert_name] = (metrics, per_motion)
    finally:
        if split == "train" and saved_valid is not None:
            for ctx in trainer.experts.values():
                ctx.wrapper.env.end_eval()
            trainer.valid_experts = saved_valid

    return results


def mine_split(trainer, split, split_cfg, run_cfg, checkpoint_path):
    """Eval one split and save results."""
    mean_mpjpe = split_cfg.get("mean_mpjpe", None)
    max_mpjpe = split_cfg.get("max_mpjpe", None)
    not_success = split_cfg.get("not_success", False)

    logger.info(f"{'=' * 60}")
    logger.info(f"Mining: {split}")
    logger.info(f"  mean_mpjpe={mean_mpjpe}, max_mpjpe={max_mpjpe}, not_success={not_success}")
    logger.info(f"{'=' * 60}")

    detailed = _evaluate_all_experts(trainer, split)

    for expert_name, (metrics, per_motion) in detailed.items():
        negatives = filter_negatives(
            per_motion,
            mean_mpjpe=mean_mpjpe,
            max_mpjpe=max_mpjpe,
            not_success=not_success,
        )

        log_summary(expert_name, split, per_motion, negatives)

        criteria = {
            "mean_mpjpe": mean_mpjpe,
            "max_mpjpe": max_mpjpe,
            "not_success": not_success,
        }
        multi_expert = len(detailed) > 1
        txt_path = save_results(
            per_motion=per_motion,
            negatives=negatives,
            output_dir=run_cfg.output_dir,
            output_name=run_cfg.output_name,
            split=split,
            expert_name=expert_name if multi_expert else "",
            checkpoint_path=checkpoint_path,
            epoch=trainer.epoch,
            criteria=criteria,
        )
        logger.info(f"  → {txt_path}")


def hydra_main():
    """Mine mode: eval + filter via Hydra config."""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../../cfg", config_name="config", version_base=None)
    def _main(cfg: DictConfig) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        from arnold.trainer import ArnoldTrainer

        run_cfg = cfg.run
        checkpoint_path = cfg.get("resume_checkpoint", None)
        if checkpoint_path is None:
            raise ValueError("resume_checkpoint is required.")

        logger.info("=" * 60)
        logger.info("Hard Negative Mining")
        logger.info("=" * 60)
        logger.info(f"  checkpoint:   {checkpoint_path}")
        logger.info(f"  mine_train:   {run_cfg.mine_train}")
        logger.info(f"  mine_test:    {run_cfg.mine_test}")
        logger.info(f"  train:        {dict(run_cfg.train)}")
        logger.info(f"  test:         {dict(run_cfg.test)}")
        logger.info(f"  output_name:  {run_cfg.output_name}")
        logger.info(f"  output_dir:   {run_cfg.output_dir}")

        cfg.use_wandb = False
        cfg.no_log = True

        trainer = ArnoldTrainer(cfg, device=cfg.device)

        if run_cfg.mine_test:
            mine_split(trainer, "valid", run_cfg.test, run_cfg, checkpoint_path)
        if run_cfg.mine_train:
            mine_split(trainer, "train", run_cfg.train, run_cfg, checkpoint_path)

        logger.info("Done!")

    _main()


# ─────────────────────────────────────────────────────────────────────
#  Refilter (no eval, just re-apply thresholds to existing JSON)
# ─────────────────────────────────────────────────────────────────────

def refilter_main():
    """Re-apply filter thresholds to an existing JSON without re-running eval."""
    parser = argparse.ArgumentParser(
        description="Refilter negatives from existing JSON"
    )
    parser.add_argument("--json", required=True, help="Path to existing *_split.json")
    parser.add_argument("--mean-mpjpe", type=float, default=None)
    parser.add_argument("--max-mpjpe", type=float, default=None)
    parser.add_argument("--not-success", action="store_true")
    parser.add_argument("--output", default=None, help="Output .txt path (default: replaces original)")
    args = parser.parse_args(sys.argv[2:])  # skip "refilter"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    with open(args.json, "r") as f:
        data = json.load(f)

    all_results = data["all_results"]
    meta = data["meta"]

    negatives = filter_negatives(
        all_results,
        mean_mpjpe=args.mean_mpjpe,
        max_mpjpe=args.max_mpjpe,
        not_success=args.not_success,
    )

    split = meta.get("split", "unknown")
    expert = meta.get("expert_name", "")
    log_summary(expert or "myo", split, all_results, negatives)

    # Save txt
    if args.output is None:
        txt_path = args.json.replace(".json", ".txt")
    else:
        txt_path = args.output
    save_txt(negatives, len(all_results), txt_path)

    # Update JSON meta
    meta.update({
        "mean_mpjpe": args.mean_mpjpe,
        "max_mpjpe": args.max_mpjpe,
        "not_success": args.not_success,
        "negatives_count": len(negatives),
    })
    json_path = txt_path.replace(".txt", ".json")
    save_json(all_results, negatives, meta, json_path)

    logger.info(f"  → {txt_path}")
    logger.info(f"  → {json_path}")


# ─────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "refilter":
        refilter_main()
    else:
        hydra_main()
