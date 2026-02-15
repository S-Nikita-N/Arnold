#!/usr/bin/env python3
"""
Скрипт обучения Arnold.

Режим определяется автоматически по весам лоссов:
  OBC:      ppo_weight=0,  imitation_weight>0
  OBC-PPO:  ppo_weight>0,  imitation_weight>0
  PPO:      ppo_weight>0,  imitation_weight=0

Примеры:

    # OBC дистилляция из Kinesis
    poetry run python -m arnold.train_arnold exp_name=obc_kinesis

    # PPO на MyoHuman, с переносом знаний из OBC чекпоинта
    poetry run python -m arnold.train_arnold \\
        env=myohuman \\
        'run.experts=[{type: myohuman, config_path: null, checkpoint_epoch: 0}]' \\
        resume_checkpoint=/path/to/model_best_ep_length.pth \\
        learning.ppo_weight=1.0 \\
        learning.imitation_weight=0 \\
        learning.entropy_weight=0.01 \\
        exp_name=ppo_myohuman_transfer
"""

import os
import sys
import logging
import warnings

from pathlib import Path
from dotenv import load_dotenv

import hydra
from omegaconf import DictConfig, OmegaConf
from arnold.trainer import ArnoldTrainer

# Загрузка .env из корня репо
_repo_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_repo_root / ".env")

_http_proxy = os.getenv("HTTP_PROXY")
_https_proxy = os.getenv("HTTPS_PROXY")
if _http_proxy:
    os.environ["HTTP_PROXY"] = _http_proxy
if _https_proxy:
    os.environ["HTTPS_PROXY"] = _https_proxy

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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

    # Определяем режим
    if cfg.learning.imitation_weight > 0 and cfg.learning.ppo_weight > 0:
        mode_str = "OBC-PPO"
    elif cfg.learning.imitation_weight > 0:
        mode_str = "OBC"
    else:
        mode_str = "PPO"

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
