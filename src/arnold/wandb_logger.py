"""
WandB Logger для Arnold OBC Training.
"""

import os
import logging
import psutil
import numpy as np
import torch
import wandb
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
from arnold.logger import OBCLogger


logger = logging.getLogger(__name__)


class WandbLogger:
    """
    Обёртка над wandb для Arnold training.
    
    Usage:
        wandb_logger = WandbLogger(cfg)
        wandb_logger.log(metrics, step=epoch)
        wandb_logger.finish()
    """
    
    def __init__(
        self,
        cfg: DictConfig,
    ):
        """
        Args:
            cfg: Hydra конфигурация
        """
        self.cfg = cfg
        
        self._init_wandb()
    
    def _init_wandb(self) -> None:
        """Инициализирует wandb run."""
        # Convert config to dict
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        
        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.exp_name,
            notes=self.cfg.notes,
            config=config_dict,
            dir=self.cfg.run.output_dir,
            resume="allow",
        )
        
        logger.info(f"WandB initialized: {wandb.run.name} ({wandb.run.id})")
    
    def log(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = "",
    ) -> None:
        """
        Логирует метрики в wandb.
        
        Args:
            metrics: Dict с метриками
            step: Номер шага/эпохи
            prefix: Префикс для ключей
        """
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        wandb.log(metrics, step=step)
    
    def log_train(
        self,
        epoch: int,
        obc_logger: OBCLogger,
        total_steps: int,
    ) -> None:
        """
        Логирует метрики тренировки (по аналогии с Kinesis).
        
        Args:
            epoch: Номер эпохи
            obc_logger: OBCLogger с метриками сэмплирования
            total_steps: Общее число шагов
        """
        # Resource usage
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
        
        # Base metrics
        log_dict = {
            # Rewards
            "reward/avg_episode": obc_logger.avg_episode_reward,
            "reward/avg_step": obc_logger.avg_reward,
            "reward/min": obc_logger.min_reward,
            "reward/max": obc_logger.max_reward,
            "reward/min_episode": obc_logger.min_episode_reward,
            "reward/max_episode": obc_logger.max_episode_reward,
            
            # Episodes
            "episode/avg_length": obc_logger.avg_episode_len,
            "episode/count": obc_logger.num_episodes,
            
            # Losses (из update_params)
            "loss/ppo": obc_logger.ppo_loss,
            "loss/imitation": obc_logger.imitation_loss,
            "loss/value": obc_logger.value_loss,
            "loss/entropy": obc_logger.entropy_loss,
            "loss/total": obc_logger.total_loss,
            
            # Timing
            "time/sample": obc_logger.sample_time,
            "time/update": obc_logger.update_time,
            
            # Resources
            "resource/cpu_mem_gb": cpu_mem,
            "resource/gpu_mem_gb": gpu_mem,
            
            # Progress
            "progress/epoch": epoch,
            "progress/total_steps": total_steps,
            "progress/batch_steps": obc_logger.num_steps,
        }
        
        # Reward components from info_dict
        for k, v in obc_logger.info_dict.items():
            if v:
                log_dict[f"reward_component/{k}"] = np.mean(v)
        
        wandb.log(log_dict, step=epoch)
    
    def log_eval(
        self,
        epoch: int,
        eval_metrics: Dict[str, float],
    ) -> None:
        """
        Логирует метрики evaluation.
        
        Args:
            epoch: Номер эпохи
            eval_metrics: Dict с метриками
        """
        wandb.log(eval_metrics, step=epoch)
    
    def log_config(self) -> None:
        """Логирует конфигурацию как artifact."""
        # Save config to file
        config_path = os.path.join(self.cfg.run.output_dir, "config.yaml")
        OmegaConf.save(self.cfg, config_path)
        
        # Log as artifact
        artifact = wandb.Artifact("config", type="config")
        artifact.add_file(config_path)
        wandb.log_artifact(artifact)
    
    def finish(self) -> None:
        """Завершает wandb run."""
        wandb.finish()
        logger.info("WandB run finished")
    
    def watch(self, model: torch.nn.Module, log_freq: int = 100) -> None:
        """
        Включает отслеживание градиентов модели.
        
        Args:
            model: PyTorch модель
            log_freq: Частота логирования (в batches)
        """
        wandb.watch(model, log="gradients", log_freq=log_freq)
    
    @property
    def run_dir(self) -> Optional[str]:
        """Возвращает директорию wandb run."""
        return wandb.run.dir
    
    @property
    def run_name(self) -> Optional[str]:
        """Возвращает имя wandb run."""
        return wandb.run.name
