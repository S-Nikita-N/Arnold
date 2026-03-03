#!/usr/bin/env python3
"""
Скрипт обучения Arnold.

Поддерживает single- и multi-expert обучение.
Per-expert режим определяется по loss-весам каждого эксперта.

Примеры:

    # Single expert — OBC дистилляция из Kinesis
    poetry run python -m arnold.train_arnold \
        '+run/experts@run.experts.kin=kinesis' \
        exp_name=obc_kinesis

    # Single expert — PPO на MyoHuman
    poetry run python -m arnold.train_arnold \
        '+run/experts@run.experts.myo=myohuman' \
        run.experts.myo.simple=true \
        run.experts.myo.learning.ppo_weight=1.0 \
        run.experts.myo.learning.imitation_weight=0 \
        run.experts.myo.learning.entropy_weight=0.0001 \
        run.experts.myo.learning.value_weight=0.5 \
        run.num_threads=15 \
        exp_name=ppo_myohuman

    # Multi-expert — MyoHuman (PPO) + Kinesis (OBC)
    poetry run python -m arnold.train_arnold \
        device=cuda \
        '+run/experts@run.experts.myo=myohuman' \
        '+run/experts@run.experts.kin=kinesis' \
        \
        run.experts.myo.simple=true \
        run.experts.myo.num_threads=10 \
        run.experts.myo.min_batch_size=20480 \
        run.experts.myo.learning.ppo_weight=1.0 \
        run.experts.myo.learning.imitation_weight=0 \
        run.experts.myo.learning.entropy_weight=0.0001 \
        run.experts.myo.learning.value_weight=0.5 \
        \
        run.experts.kin.num_threads=5 \
        run.experts.kin.min_batch_size=10240 \
        run.experts.kin.learning.ppo_weight=0.0 \
        run.experts.kin.learning.imitation_weight=1.0 \
        run.experts.kin.learning.entropy_weight=0.0 \
        run.experts.kin.learning.value_weight=0.0 \
        \
        learning.batch_size=256 \
        learning.learning_rate=1e-4 \
        learning.grad_clip=15 \
        learning.opt_num_epochs=5 \
        learning.tokenizer_granularity=per_spec \
        \
        resume_checkpoint=path/to/pretrained.pth \
        exp_name=multi_myo_kin
"""

import os
import sys
import hydra
import logging
import warnings

from pathlib import Path
from omegaconf import DictConfig, OmegaConf

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from arnold.trainer import ArnoldTrainer

_repo_root = Path(__file__).resolve().parent.parent.parent
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")


def setup_logging(output_dir: str) -> logging.Logger:
    """Настраивает логирование."""
    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "training.log")),
        ]
    )
    return logging.getLogger(__name__)


@hydra.main(config_path="../../cfg", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = cfg.run.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)

    # Определяем режим по экспертам
    experts_cfg = cfg.run.experts
    expert_names = list(experts_cfg.keys()) if experts_cfg else []

    if len(expert_names) > 1:
        mode_str = f"MULTI-EXPERT ({', '.join(expert_names)})"
    elif len(expert_names) == 1:
        entry = list(experts_cfg.values())[0]
        ppo_w = entry.get("learning", {}).get("ppo_weight", cfg.learning.ppo_weight)
        im_w = entry.get("learning", {}).get("imitation_weight", cfg.learning.imitation_weight)
        if ppo_w is None:
            ppo_w = cfg.learning.ppo_weight
        if im_w is None:
            im_w = cfg.learning.imitation_weight
        if im_w > 0 and ppo_w > 0:
            mode_str = "OBC-PPO"
        elif im_w > 0:
            mode_str = "OBC"
        else:
            mode_str = "PPO"
    else:
        mode_str = "NO EXPERTS"

    logger.info("=" * 60)
    logger.info(f"Arnold Training — {mode_str}")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    config_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(cfg, config_path)

    trainer = ArnoldTrainer(cfg, device=cfg.device)
    trainer.optimize_policy()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
