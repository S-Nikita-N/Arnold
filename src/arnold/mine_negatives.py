#!/usr/bin/env python3
"""
Hard Negative Mining — оценка политики на всех motion и отбор провальных.

Запускает vectorized (или sequential) evaluation на train/test сплите,
собирает per-motion метрики и фильтрует по заданным критериям.

Результат:
  - {output_name}_{split}.txt — motion_id по одному на строку (для MyoLegsIm)
  - {output_name}_{split}.json — полная мета + per-motion метрики

Использование:

    # Mine from both train and test (default)
    poetry run python -m arnold.mine_negatives \
        run=mine_negatives \
        '+run/experts@run.experts.myo=myohuman' \
        resume_checkpoint=runs/exp/model.pth

    # Only test, custom thresholds
    poetry run python -m arnold.mine_negatives \
        run=mine_negatives \
        '+run/experts@run.experts.myo=myohuman' \
        resume_checkpoint=runs/exp/model.pth \
        run.mine_train=false \
        run.mean_mpjpe=0.08 \
        run.max_mpjpe=0.15

    # Дообучение на негативах (train + valid фильтруются)
    poetry run python -m arnold.train_arnold \
        '+run/experts@run.experts.myo=myohuman' \
        'run.experts.myo.overrides=["run.motion_ids_file=/path/negatives_train.txt"]' \
        resume_checkpoint=runs/exp/model.pth

Критерии негатива (OR — любой из):
    - not_success: success == False (ранняя терминация)
    - mean_mpjpe: mean MPJPE > threshold
    - max_mpjpe: max per-frame MPJPE > threshold
"""

import os
import sys
import json
import logging

import numpy as np
import hydra
from omegaconf import DictConfig

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from arnold.trainer import ArnoldTrainer


logger = logging.getLogger(__name__)


def filter_negatives(
    results: list,
    mean_mpjpe: float = None,
    max_mpjpe: float = None,
    not_success: bool = False,
) -> list:
    """
    Фильтрует per-motion results по критериям негатива (OR).

    Критерии:
        - not_success: success == False
        - mean_mpjpe: mpjpe > threshold
        - max_mpjpe: max_mpjpe > threshold
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


def save_results(
    per_motion: list,
    negatives: list,
    output_dir: str,
    output_name: str,
    split: str,
    expert_name: str,
    checkpoint_path: str,
    epoch: int,
    run_cfg,
) -> str:
    """Saves txt (motion IDs) + json (full meta). Returns txt path."""
    base = f"{output_name}_{split}"
    if expert_name:
        base = f"{base}_{expert_name}"
    txt_path = os.path.join(output_dir, f"{base}.txt")
    json_path = os.path.join(output_dir, f"{base}.json")

    os.makedirs(output_dir, exist_ok=True)

    # ── txt: plain motion IDs ────────────────────────────────────────
    neg_ids = sorted(int(r["motion_id"]) for r in negatives)
    with open(txt_path, "w") as f:
        f.write(f"# {len(neg_ids)} negatives / {len(per_motion)} total\n")
        for mid in neg_ids:
            f.write(f"{mid}\n")

    # ── json: full meta + per-motion ─────────────────────────────────
    meta = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "epoch": epoch,
        "expert_name": expert_name,
        "mean_mpjpe": run_cfg.mean_mpjpe,
        "max_mpjpe": run_cfg.max_mpjpe,
        "not_success": run_cfg.not_success,
        "total_motions": len(per_motion),
        "negatives_count": len(negatives),
    }
    output = {
        "meta": meta,
        "negatives": negatives,
        "all_results": per_motion,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return txt_path


def log_group_stats(items: list, label: str) -> None:
    """Log stats for a group of per-motion results."""
    if not items:
        logger.info(f"  {label}: (empty)")
        return
    n = len(items)
    mpjpes = [r["mpjpe"] for r in items if r.get("mpjpe") is not None]
    max_mpjpes = [r["max_mpjpe"] for r in items if r.get("max_mpjpe") is not None]
    coverages = [r["frame_coverage"] for r in items if r.get("frame_coverage") is not None]
    rewards = [r["reward"] for r in items if r.get("reward") is not None]
    successes = [r["success"] for r in items if r.get("success") is not None]

    parts = [f"{label} ({n})"]
    if successes:
        parts.append(f"success={np.mean(successes):.1%}")
    if mpjpes:
        parts.append(f"mean_mpjpe={np.mean(mpjpes)*1000:.1f}mm")
    if max_mpjpes:
        parts.append(f"max_mpjpe={np.mean(max_mpjpes)*1000:.1f}mm")
    if coverages:
        parts.append(f"frame_cov={np.mean(coverages):.3f}")
    if rewards:
        parts.append(f"reward={np.mean(rewards):.1f}±{np.std(rewards):.1f}")
    logger.info(f"  {' | '.join(parts)}")


def log_summary(expert_name: str, split: str, per_motion: list, negatives: list) -> None:
    """Log stats for negatives and positives."""
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


def mine_split(
    trainer: ArnoldTrainer,
    split: str,
    run_cfg,
    checkpoint_path: str,
) -> None:
    """Run mining on one split (train or valid)."""
    logger.info(f"{'='*60}")
    logger.info(f"Mining: {split}")
    logger.info(f"{'='*60}")

    detailed = trainer.evaluate_detailed(split=split)

    for expert_name, (metrics, per_motion) in detailed.items():
        negatives = filter_negatives(
            per_motion,
            mean_mpjpe=run_cfg.mean_mpjpe,
            max_mpjpe=run_cfg.max_mpjpe,
            not_success=run_cfg.not_success,
        )

        log_summary(expert_name, split, per_motion, negatives)

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
            run_cfg=run_cfg,
        )
        logger.info(f"  → {txt_path}")


@hydra.main(config_path="../../cfg", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    run_cfg = cfg.run
    checkpoint_path = cfg.get("resume_checkpoint", None)
    if checkpoint_path is None:
        raise ValueError(
            "resume_checkpoint is required. "
            "Pass resume_checkpoint=/path/to/model.pth"
        )

    logger.info("=" * 60)
    logger.info("Hard Negative Mining")
    logger.info("=" * 60)
    logger.info(f"  checkpoint:   {checkpoint_path}")
    logger.info(f"  mine_train:   {run_cfg.mine_train}")
    logger.info(f"  mine_test:    {run_cfg.mine_test}")
    logger.info(f"  mean_mpjpe:   {run_cfg.mean_mpjpe}")
    logger.info(f"  max_mpjpe:    {run_cfg.max_mpjpe}")
    logger.info(f"  not_success:  {run_cfg.not_success}")
    logger.info(f"  output_name:  {run_cfg.output_name}")
    logger.info(f"  output_dir:   {run_cfg.output_dir}")

    # ── Disable wandb ────────────────────────────────────────────────
    cfg.use_wandb = False
    cfg.no_log = True

    # ── Create trainer ───────────────────────────────────────────────
    trainer = ArnoldTrainer(cfg, device=cfg.device)

    # ── Mine splits ──────────────────────────────────────────────────
    if run_cfg.mine_test:
        mine_split(trainer, "valid", run_cfg, checkpoint_path)
    if run_cfg.mine_train:
        mine_split(trainer, "train", run_cfg, checkpoint_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()


# poetry run python -m arnold.mine_negatives \
#     device=mps \
#     run=mine_negatives \
#     '+run/experts@run.experts.myo=myohuman' \
#     resume_checkpoint=/Users/nikita/Projects/diploma/arnold/data/trained_models/norway/model_best_ep_length_negative_1_not-finished.pth \
#     run.mean_mpjpe=0.08 \
#     run.max_mpjpe=0.15 \
#     run.experts.myo.num_threads=8 \
#     learning.policy=lattice \
#     learning.transfer_mode=false    
