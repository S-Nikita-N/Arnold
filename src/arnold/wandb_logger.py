"""
WandB Logger для Arnold OBC Training.

Поддерживает multi-expert режим: метрики логируются с per-expert префиксами.
Пример: myohuman/reward/avg_episode, kinesis/loss/imitation
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
        wandb_logger.log_train_multi(epoch, expert_loggers, diagnostics, total_steps)
        wandb_logger.finish()
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._init_wandb()

    def _init_wandb(self) -> None:
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)

        run_id = self.cfg.get("wandb_run_id")
        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.exp_name,
            notes=self.cfg.notes,
            config=config_dict,
            dir=self.cfg.run.output_dir,
            id=run_id,
            resume="must" if run_id else "allow",
        )
        logger.info(f"WandB initialized: {wandb.run.name} ({wandb.run.id})")

    def log(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = "",
    ) -> None:
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        wandb.log(metrics, step=step)

    def log_train(
        self,
        epoch: int,
        expert_loggers: Dict[str, OBCLogger],
        all_diagnostics: Dict[str, Dict[str, float]],
        total_steps: int,
    ) -> None:
        """
        Логирует метрики тренировки для всех экспертов.

        Формат ключей: {expert_name}/{category}/{metric}
        Пример: myohuman/reward/avg_episode, kinesis/loss/imitation
        """
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / 1024 / 1024 / 1024
        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024

        log_dict = {
            "resource/cpu_mem_gb": cpu_mem,
            "resource/gpu_mem_gb": gpu_mem,
            "progress/epoch": epoch,
            "progress/total_steps": total_steps,
        }

        for expert_name, obc_logger in expert_loggers.items():

            # Rewards
            log_dict[f"{expert_name}/reward/avg_episode"] = obc_logger.avg_episode_reward
            log_dict[f"{expert_name}/reward/avg_step"] = obc_logger.avg_reward
            log_dict[f"{expert_name}/reward/min"] = obc_logger.min_reward
            log_dict[f"{expert_name}/reward/max"] = obc_logger.max_reward
            log_dict[f"{expert_name}/reward/min_episode"] = obc_logger.min_episode_reward
            log_dict[f"{expert_name}/reward/max_episode"] = obc_logger.max_episode_reward

            # Episodes
            log_dict[f"{expert_name}/episode/avg_length"] = obc_logger.avg_episode_len
            log_dict[f"{expert_name}/episode/count"] = obc_logger.num_episodes

            # Losses
            log_dict[f"{expert_name}/loss/ppo"] = obc_logger.ppo_loss
            log_dict[f"{expert_name}/loss/imitation"] = obc_logger.imitation_loss
            log_dict[f"{expert_name}/loss/value"] = obc_logger.value_loss
            log_dict[f"{expert_name}/loss/entropy"] = obc_logger.entropy_loss
            log_dict[f"{expert_name}/loss/total"] = obc_logger.total_loss

            # Timing
            log_dict[f"{expert_name}/time/sample"] = obc_logger.sample_time
            log_dict[f"{expert_name}/time/update"] = obc_logger.update_time

            # Steps
            log_dict[f"{expert_name}/progress/batch_steps"] = obc_logger.num_steps

            # Reward components & eval metrics from sampling rollouts
            eval_keys = {"mpjpe", "frame_coverage", "success"}
            for k, v in obc_logger.info_dict.items():
                if v:
                    prefix = "train_rollout" if k in eval_keys else "reward_component"
                    log_dict[f"{expert_name}/{prefix}/{k}"] = np.mean(v)

            # PPO diagnostics
            diag = all_diagnostics.get(expert_name, {})
            for k, v in diag.items():
                if isinstance(v, (int, float)):
                    log_dict[f"{expert_name}/diag/{k}"] = v

            # Явный лог im_contrib чтобы сразу видеть баланс с PPO
            if "imitation_loss" in diag and "imitation_loss_per_sample" in diag:
                log_dict[f"{expert_name}/loss/imitation_batch"] = diag["imitation_loss"]
                log_dict[f"{expert_name}/loss/imitation_per_sample"] = diag["imitation_loss_per_sample"]

        wandb.log(log_dict, step=epoch)

    def log_eval(
        self,
        epoch: int,
        eval_metrics: Dict[str, float],
    ) -> None:
        wandb.log(eval_metrics, step=epoch)

    def log_config(self) -> None:
        config_path = os.path.join(self.cfg.run.output_dir, "config.yaml")
        OmegaConf.save(self.cfg, config_path)
        artifact = wandb.Artifact("config", type="config")
        artifact.add_file(config_path)
        wandb.log_artifact(artifact)

    def finish(self) -> None:
        wandb.finish()
        logger.info("WandB run finished")

    def watch(self, model: torch.nn.Module, log_freq: int = 100) -> None:
        wandb.watch(model, log="gradients", log_freq=log_freq)

    @property
    def run_dir(self) -> Optional[str]:
        return wandb.run.dir

    @property
    def run_name(self) -> Optional[str]:
        return wandb.run.name
