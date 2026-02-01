#!/usr/bin/env python3
"""
Скрипт для запуска OBC обучения Arnold с Hydra конфигурацией.

Usage:
    # Базовый запуск (из директории arnold)
    python train_arnold.py
    
    # С overrides
    python train_arnold.py learning.max_epochs=2000 learning.batch_size=128
    
    # С wandb
    python train_arnold.py use_wandb=true exp_name=my_exp
    
    # Продолжение обучения
    python train_arnold.py epoch=-1
    
    # Multiprocessing
    python train_arnold.py run.num_threads=4
    
    # Указать эксперта
    python train_arnold.py run.expert.config_path=/path/to/kinesis/cfg
"""

import os
import sys
import logging
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf
from arnold.obc_trainer import OBCTrainer

# Игнорируем SyntaxWarning про invalid escape sequence в docstrings Kinesis
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


@hydra.main(config_path="cfg", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra config."""
    
    # Setup output directory
    output_dir = cfg.run.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("=" * 60)
    logger.info("Arnold OBC Training")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Save full config
    config_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(cfg, config_path)
    logger.info(f"Config saved to {config_path}")
    
    # Create trainer
    trainer = OBCTrainer(cfg, device=cfg.device)
    
    # Train
    trainer.optimize_policy()
    
    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate(num_episodes=10)
    logger.info(f"Final evaluation: {eval_results}")
    
    logger.info("Training completed!")


# poetry run python -m myohuman.arnold.train_arnold \
#     use_wandb=true \
#     exp_name=obc_v2_run_2 \
#     device=mps \
#     learning.learning_rate=1e-3 \
#     learning.min_batch_size=5120 \
#     run.num_threads=5

# poetry run python -m myohuman.arnold.train_arnold \
#     use_wandb=true \
#     exp_name=obc_run_A100_80GB_1 \
#     device=cuda \
#     learning.learning_rate=1e-3 \
#     learning.min_batch_size=25120 \
#     run.num_threads=25

if __name__ == "__main__":
    main()
