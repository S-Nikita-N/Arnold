#!/usr/bin/env python3
"""
Hard Negative Mining — оценка политики на всех motion и отбор провальных.

Два режима:
  1. mine   — eval + filter + save (default)
  2. refilter — пересчитать фильтр из существующего JSON без повторного eval

Результат (split = train | valid — valid соответствует eval/val):
  - {output_name}_{split}.json — meta + ключ \"frames\": все мотии
    eval (и «хорошие», и «плохие»)
  - {output_name}_{split}_negatives.txt — motion_id негативов
    (для motion_ids_file и т.п.)
  - {output_name}_{split}_positives.txt — комплемент по тем же порогам

  По умолчанию output_name=eval → eval_train.json, eval_valid.json и пары txt.

Критерии негатива задаются раздельно для train и test:
  run.train.mean_mpjpe=0.05   (строже — модель видела эти motions)
  run.test.mean_mpjpe=0.08    (мягче — unseen data)

Использование:

    # Mine (eval + filter)
    uv run python -m arnold.mine_negatives \
        run=mine_negatives \
        '+run/experts@run.experts.myo=myohuman' \
        resume_checkpoint=runs/exp/model.pth

    # Refilter: JSON с ключом "frames", перезапись того же
    #           .json и *_negatives/_positives.txt
    uv run python -m arnold.mine_negatives refilter \
        --json data/trained_models/exp/eval_valid.json \
        --mean-mpjpe 0.06 --not-success
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


########################################
#                Filter                #
########################################


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
        if (
            mean_mpjpe is not None
            and r.get("mpjpe") is not None
            and r["mpjpe"] > mean_mpjpe
        ):
            is_negative = True
        if (
            max_mpjpe is not None
            and r.get("max_mpjpe") is not None
            and r["max_mpjpe"] > max_mpjpe
        ):
            is_negative = True

        if is_negative:
            negatives.append(r)

    return negatives


def derive_positives(per_motion: list, negatives: list) -> list:
    """Комплемент к негативам по motion_id (один motion — одна запись)."""
    neg_ids = {int(r["motion_id"]) for r in negatives}
    return [r for r in per_motion if int(r["motion_id"]) not in neg_ids]


########################################
#                 Save                 #
########################################


def save_motion_id_txt(items: list, total: int, path: str, label: str) -> None:
    """Построчно motion_id; первая строка — счётчик
    (label: negatives | positives)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ids_ = sorted(int(r["motion_id"]) for r in items)
    with open(path, "w") as f:
        f.write(f"# {len(ids_)} {label} / {total} total\n")
        for mid in ids_:
            f.write(f"{mid}\n")


def save_eval_json(frames: list, meta: dict, path: str) -> None:
    """Полный eval: meta + frames (все мотии)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    output = {"meta": meta, "frames": frames}
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
) -> tuple[str, str, str]:
    """Пишет eval json + два txt. Возвращает (neg_txt, pos_txt, json_path)."""
    base = f"{output_name}_{split}"
    if expert_name:
        base = f"{base}_{expert_name}"
    stem = os.path.join(output_dir, base)
    neg_path = f"{stem}_negatives.txt"
    pos_path = f"{stem}_positives.txt"
    json_path = f"{stem}.json"

    positives = derive_positives(per_motion, negatives)
    total = len(per_motion)
    save_motion_id_txt(negatives, total, neg_path, "negatives")
    save_motion_id_txt(positives, total, pos_path, "positives")

    meta = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "epoch": epoch,
        "expert_name": expert_name,
        "total_motions": total,
        "negatives_count": len(negatives),
        "positives_count": len(positives),
        **criteria,
    }
    save_eval_json(per_motion, meta, json_path)

    return neg_path, pos_path, json_path


########################################
#               Logging                #
########################################


def log_group_stats(items: list, label: str) -> None:
    if not items:
        logger.info(f"  {label}: (empty)")
        return
    mpjpes = [r["mpjpe"] for r in items if r.get("mpjpe") is not None]
    max_mpjpes = [
        r["max_mpjpe"] for r in items if r.get("max_mpjpe") is not None
    ]
    coverages = [
        r["frame_coverage"]
        for r in items
        if r.get("frame_coverage") is not None
    ]
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


def log_summary(
    expert_name: str, split: str, per_motion: list, negatives: list
) -> None:
    total = len(per_motion)
    n_neg = len(negatives)
    positives = derive_positives(per_motion, negatives)

    logger.info(
        f"[{expert_name}] {split}: "
        f"{n_neg} negatives + {len(positives)} positives = {total} total"
    )
    log_group_stats(negatives, "negatives")
    log_group_stats(positives, "positives")


########################################
#         Mine (eval + filter)         #
########################################


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

            if trainer.vectorized_eval and trainer.device.type in (
                "cuda",
                "mps",
            ):
                metrics, per_motion = trainer._evaluate_expert_vectorized(
                    expert_name,
                    ctx,
                    wrapper,
                    return_per_motion=True,
                )
            else:
                metrics, per_motion = trainer.evaluate_expert(
                    expert_name,
                    ctx,
                    wrapper,
                    return_per_motion=True,
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
    logger.info(
        f"  mean_mpjpe={mean_mpjpe}, max_mpjpe={max_mpjpe}, "
        f"not_success={not_success}"
    )
    logger.info(f"{'=' * 60}")

    detailed = _evaluate_all_experts(trainer, split)

    for expert_name, (_metrics, per_motion) in detailed.items():
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
        neg_path, pos_path, json_path = save_results(
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
        logger.info(f"  → {neg_path}")
        logger.info(f"  → {pos_path}")
        logger.info(f"  → {json_path}")


def hydra_main():
    """Mine mode: eval + filter via Hydra config."""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(
        config_path="../../cfg", config_name="config", version_base=None
    )
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
            mine_split(
                trainer, "train", run_cfg.train, run_cfg, checkpoint_path
            )

        logger.info("Done!")

    _main()


########################################
#               Refilter               #
########################################


def refilter_main():
    """Перефильтрация порогами без eval:
    JSON с ключом \"frames\" → тот же json + txt."""
    parser = argparse.ArgumentParser(
        description="Refilter from eval JSON (required key: frames)",
    )
    parser.add_argument(
        "--json",
        required=True,
        dest="json_path",
        help="Путь к eval_*.json (обязательно поле frames)",
    )
    parser.add_argument("--mean-mpjpe", type=float, default=None)
    parser.add_argument("--max-mpjpe", type=float, default=None)
    parser.add_argument("--not-success", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Явный путь к *_negatives.txt (позитивы — "
            "*_positives.txt рядом; json по умолчанию всё равно --json)"
        ),
    )
    args = parser.parse_args(sys.argv[2:])  # skip "refilter"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    with open(args.json_path) as f:
        data = json.load(f)

    if "frames" not in data:
        raise ValueError('Eval JSON must contain a top-level "frames" array.')
    frames = data["frames"]
    meta = data.get("meta", {})

    negatives = filter_negatives(
        frames,
        mean_mpjpe=args.mean_mpjpe,
        max_mpjpe=args.max_mpjpe,
        not_success=args.not_success,
    )

    split = meta.get("split", "unknown")
    expert = meta.get("expert_name", "")
    log_summary(expert or "myo", split, frames, negatives)

    inp = os.path.abspath(args.json_path)
    json_out = inp

    if args.output is None:
        stem = os.path.splitext(inp)[0]
        neg_path = f"{stem}_negatives.txt"
        pos_path = f"{stem}_positives.txt"
    else:
        neg_path = os.path.abspath(args.output)
        dire = os.path.dirname(neg_path)
        base = os.path.basename(neg_path)
        if base.endswith("_negatives.txt"):
            pos_path = os.path.join(
                dire, base.replace("_negatives.txt", "_positives.txt")
            )
        else:
            stem, _ext = os.path.splitext(neg_path)
            pos_path = f"{stem}_positives.txt"

    total = len(frames)
    save_motion_id_txt(negatives, total, neg_path, "negatives")
    positives = derive_positives(frames, negatives)
    save_motion_id_txt(positives, total, pos_path, "positives")

    meta = dict(meta)
    meta.update(
        {
            "mean_mpjpe": args.mean_mpjpe,
            "max_mpjpe": args.max_mpjpe,
            "not_success": args.not_success,
            "negatives_count": len(negatives),
            "positives_count": len(positives),
            "total_motions": total,
        }
    )
    save_eval_json(frames, meta, json_out)

    logger.info(f"  → {neg_path}")
    logger.info(f"  → {pos_path}")
    logger.info(f"  → {json_out}")


########################################
#             Entry point              #
########################################

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "refilter":
        refilter_main()
    else:
        hydra_main()
