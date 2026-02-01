"""
On-Policy Behavior Cloning (OBC) Trainer для Arnold.

Обучает TransformerPolicy имитировать эксперта Kinesis используя OBC:
1. Student (Arnold) взаимодействует со средой
2. Expert (Kinesis) предоставляет target actions
3. Loss = PPO_surrogate + MSE(student_action, expert_action) + value_loss

Поддерживает многопроцессную сборку траекторий по аналогии с Kinesis Agent.
"""

import os
import math
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as multiprocessing
import warnings

from omegaconf import DictConfig
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from arnold.torch_model.transformer_policy import TransformerPolicy
from arnold.torch_model.sensorimotor_vocabulary import SensorimotorVocabulary
from arnold.observation_parser import ObservationParser
from arnold.experts.kinesis_wrapper import KinesisWrapper
from arnold.memory import OBCMemory, OBCBatch
from arnold.logger import OBCLogger
from arnold.wandb_logger import WandbLogger
from arnold.learning_utils import to_test, to_cpu, optimizer_to

# Игнорируем SyntaxWarning про invalid escape sequence в docstrings Kinesis
# (появляется при импорте модулей Kinesis в worker процессах)
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")

os.environ["OMP_NUM_THREADS"] = "1"


logger = logging.getLogger(__name__)


class OBCTrainer:
    """
    On-Policy Behavior Cloning Trainer для Arnold.
    
    Структура похожа на AgentHumanoid:
    - setup_env(): создаёт среду и парсер
    - setup_policy(): создаёт Arnold policy
    - setup_expert(): загружает эксперта Kinesis
    - sample(): собирает траектории
    - update_params(): обновляет веса
    - optimize_policy(): главный цикл обучения
    
    Usage:
        cfg = OmegaConf.load("config.yaml")
        trainer = OBCTrainer(cfg, device="cuda")
        trainer.optimize_policy()
    """
    
    def __init__(
        self,
        cfg: DictConfig,
        dtype: torch.dtype = torch.float32,
        device: str = None,
    ):
        """
        Args:
            cfg: Hydra конфигурация (см. cfg/config.yaml)
            dtype: Тип данных PyTorch
            device: Устройство (None = из конфига)
        """
        self.cfg = cfg
        self.dtype = dtype
        self.device = torch.device(device if device else cfg.device)
        
        # Architecture (из cfg.learning)
        self.history_len = cfg.learning.history_len
        self.embed_dim = cfg.learning.embed_dim
        self.ff_dim = cfg.learning.ff_dim
        self.num_heads = cfg.learning.num_heads
        self.num_layers = cfg.learning.num_layers
        self.dropout = cfg.learning.dropout
        
        # PPO/Training (из cfg.learning)
        self.batch_size = cfg.learning.batch_size
        self.learning_rate = cfg.learning.learning_rate
        self.weight_decay = cfg.learning.weight_decay
        self.gamma = cfg.learning.gamma
        self.tau = cfg.learning.tau  # GAE lambda
        self.clip_epsilon = cfg.learning.clip_epsilon
        self.opt_num_epochs = cfg.learning.opt_num_epochs
        self.grad_clip = cfg.learning.grad_clip
        self.ppo_weight = cfg.learning.ppo_weight
        self.imitation_weight = cfg.learning.imitation_weight
        self.value_weight = cfg.learning.value_weight
        self.entropy_weight = cfg.learning.entropy_weight
        self.min_batch_size = cfg.learning.min_batch_size
        self.max_epochs = cfg.learning.max_epochs
        self.use_scheduler = cfg.learning.use_scheduler
        
        # Run (из cfg.run)
        self.num_threads = cfg.run.num_threads
        self.save_frequency = cfg.run.save_frequency
        self.save_curr_frequency = cfg.run.save_curr_frequency
        self.log_frequency = cfg.run.log_frequency
        self.output_dir = cfg.run.output_dir
        self.eval_frequency = cfg.run.eval_frequency
        
        # Environment (из cfg.env)
        self.resampling_interval = cfg.env.resampling_interval
        
        # Logging
        self.use_wandb = cfg.use_wandb
        self.no_log = cfg.no_log
        self.exp_name = cfg.exp_name
        
        # Resume
        self.checkpoint_epoch = cfg.epoch
        
        # Debug mode - сохранение чекпоинтов перед каждым update_params
        self.debug_checkpoints = getattr(cfg.learning, 'debug_checkpoints', False)
        
        # State
        self.epoch = 0
        self.num_steps = 0
        
        # Best model tracking (на eval)
        self.best_eval_imitation_loss = float('inf')
        self.best_eval_episode_avg_length = 0.0
        
        # Multiprocessing Event - создаём здесь чтобы передавать явно в worker'ы
        # (при spawn на macOS глобальные переменные не шарятся между процессами)
        self.mp_done = multiprocessing.Event()
        
        # Setup
        self.setup_expert()
        self.setup_parser()
        self.setup_policy()
        self.setup_optimizer()
        
        # Load checkpoint if specified
        self.load_checkpoint(self.checkpoint_epoch)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize WandB logger
        if self.use_wandb:
            self.wandb_logger = WandbLogger(cfg)
    
    def setup_expert(self) -> None:
        """Загружает эксперта Kinesis (train и valid)."""
        logger.info("Loading Kinesis expert (train)...")
        
        self.expert = KinesisWrapper(
            cfg_path=self.cfg.run.expert.config_path,
            checkpoint_epoch=self.cfg.run.expert.checkpoint_epoch,
            device="cpu",
            mode="train",
        )

        logger.info(f"Expert loaded. Obs dim: {self.expert.obs_dim}, Action dim: {self.expert.action_dim}")
        
        # Valid expert для evaluation (загружается только если eval_frequency > 0)
        self.valid_expert = None
        if self.eval_frequency > 0:
            logger.info("Loading Kinesis expert (valid)...")
            self.valid_expert = KinesisWrapper(
                cfg_path=self.cfg.run.expert.config_path,
                checkpoint_epoch=self.cfg.run.expert.checkpoint_epoch,
                device="cpu",
                mode="valid",
            )

    def setup_parser(self) -> None:
        """Создаёт парсер."""
        logger.info("Setting up parser...")

        self.parser = ObservationParser.from_env(self.expert.env, history_len=self.history_len)

        logger.info(f"Parser: {self.parser.n_obs_elements} observation elements")
    
    def setup_policy(self) -> None:
        """Создаёт Arnold TransformerPolicy."""
        logger.info("Setting up Arnold policy...")
        
        # Vocabulary
        self.vocab = SensorimotorVocabulary(embed_dim=self.embed_dim)
        
        # Policy
        self.policy = TransformerPolicy(
            vocab=self.vocab,
            history_len=self.history_len,
            embed_dim=self.embed_dim,
            ff_dim=self.ff_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.policy.to(self.device)
        
        logger.info(f"Arnold policy created. Parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
    
    def setup_optimizer(self) -> None:
        """Создаёт оптимизатор."""
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Optional: learning rate scheduler
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_epochs,
                eta_min=self.learning_rate * 0.1,
            )
        else:
            self.scheduler = None
    
    def seed_worker(self, pid: int) -> None:
        """
        Устанавливает seed для worker процесса для рандомизации.
        
        Args:
            pid: Process ID (0 для main process)
        """
        if pid > 0:
            seed = random.randint(0, 5000) * pid + self.epoch
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def sample_worker(
        self,
        pid: int,
        queue: Optional[multiprocessing.Queue],
        mp_done: multiprocessing.Event,
        min_batch_size: int,
        expert_cfg: DictConfig,
    ) -> Optional[Tuple[OBCMemory, OBCLogger]]:
        """
        Worker функция для сбора траекторий.
        
        Args:
            pid: Process ID
            queue: Queue для передачи данных (None для main process)
            mp_done: Event для синхронизации завершения (передаётся явно для spawn)
            min_batch_size: Минимальное число шагов
            expert_cfg: Конфигурация эксперта
        
        Returns:
            (memory, logger) если queue is None, иначе None
        """
        self.seed_worker(pid)
        
        # При spawn каждый worker получает свою копию self.expert/env через pickle
        worker_parser = ObservationParser.from_env(self.expert.env, self.history_len)
        
        memory = OBCMemory()
        obc_logger = OBCLogger()
        
        if pid == 0:
            pbar = tqdm(total=min_batch_size, desc="Sampling", unit="step")
        
        try:
            while obc_logger.num_steps < min_batch_size:
                obs, info = self.expert.reset()
                worker_parser.reset(obs)
                
                for t in range(10000):
                    # Get observation for Arnold
                    obs_ts, obs_sigs = worker_parser.get_observation(torch.device("cpu"))
                    act_sigs = worker_parser.get_action_signatures()
                    
                    # Arnold forward - get action with log_prob
                    with torch.no_grad():
                        student_action, log_prob, value = self.policy.get_action(
                            obs_ts, obs_sigs, act_sigs, deterministic=False
                        )
                    
                    # Expert action
                    expert_action = self.expert.get_expert_action(obs)
                    expert_action_t = torch.from_numpy(expert_action).float()
                    
                    # Step environment with student action
                    student_action_np = student_action.squeeze(0).cpu().numpy()
                    next_obs, reward, terminated, truncated, info = self.expert.step(student_action_np)
                    done = terminated or truncated
                    
                    # Store in memory
                    memory.states.append(obs_ts.squeeze(0).cpu())
                    memory.obs_signatures.append(obs_sigs)
                    memory.action_signatures.append(act_sigs)
                    memory.student_actions.append(student_action.squeeze(0).cpu())
                    memory.expert_actions.append(expert_action_t.cpu())
                    memory.rewards.append(reward)
                    memory.values.append(value.squeeze(0).cpu())
                    memory.masks.append(0.0 if done else 1.0)
                    memory.log_probs.append(log_prob.squeeze(0).cpu())
                    
                    obc_logger.step(
                        reward=reward,
                        info=info if isinstance(info, dict) else None,
                    )
                    
                    if pid == 0:
                        pbar.update(1)

                    if done:
                        break
                    
                    # Update parser and state
                    worker_parser.update(next_obs)
                    obs = next_obs
                
                obc_logger.end_episode()
        
        except Exception as e:
            import traceback
            print(f"Worker {pid} failed: {e}")
            traceback.print_exc()
        
        finally:
            if pid == 0:
                pbar.close()
            
            if queue is not None:
                queue.put([pid, memory, obc_logger])
                mp_done.wait()
            else:
                return memory, obc_logger
    
    def sample(self, min_batch_size: int) -> Tuple[OBCBatch, OBCLogger]:
        """
        Собирает траектории для OBC обучения.
        Поддерживает многопроцессную сборку.
        
        Args:
            min_batch_size: Минимальное число шагов для сбора
        
        Returns:
            batch: OBCBatch с данными
            obc_logger: Логгер с метриками
        """
        t_start = time.time()
        
        # Сбрасываем Event перед новой эпохой (важно для повторных вызовов)
        self.mp_done.clear()
        
        # Switch to test mode
        to_test(self.policy)
        
        # Run on CPU for multiprocessing
        optimizer_to(self.optimizer, torch.device('cpu'))
        with to_cpu(self.policy):
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
                queue = multiprocessing.Queue()
                memories = [None] * self.num_threads
                loggers = [None] * self.num_threads
                
                # Spawn workers (pid > 0)
                for i in range(self.num_threads - 1):
                    worker_args = (i + 1, queue, self.mp_done, thread_batch_size, self.cfg.run.expert)
                    worker = multiprocessing.Process(
                        target=self.sample_worker,
                        args=worker_args
                    )
                    worker.start()
                
                # Main process samples (pid = 0)
                memories[0], loggers[0] = self.sample_worker(
                    0, None, self.mp_done, thread_batch_size, self.cfg.run.expert
                )
                
                # Collect from workers
                for i in range(self.num_threads - 1):
                    pid, worker_memory, worker_logger = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger
                
                # Merge memories
            merged_memory = OBCMemory()
            for mem in memories:
                if mem is not None:
                    merged_memory.states.extend(mem.states)
                    merged_memory.obs_signatures.extend(mem.obs_signatures)
                    merged_memory.action_signatures.extend(mem.action_signatures)
                    merged_memory.student_actions.extend(mem.student_actions)
                    merged_memory.expert_actions.extend(mem.expert_actions)
                    merged_memory.rewards.extend(mem.rewards)
                    merged_memory.values.extend(mem.values)
                    merged_memory.masks.extend(mem.masks)
                    merged_memory.log_probs.extend(mem.log_probs)
            
            # Merge loggers
            merged_logger = OBCLogger.merge(loggers)
        
        merged_logger.sample_time = time.time() - t_start
        
        # Signal workers to exit wait()
        self.mp_done.set()
        
        # Restore optimizer to device
        optimizer_to(self.optimizer, self.device)
        
        # Convert to batch
        batch = self._memory_to_batch(merged_memory)
        
        return batch, merged_logger
    
    def _memory_to_batch(self, memory: OBCMemory) -> OBCBatch:
        """
        Обёртка над OBCMemory.to_batch, чтобы не держать GAE-логику в трейнере.
        """
        return memory.to_batch(
            gamma=self.gamma,
            tau=self.tau,
            device=None,
        )
    
    def update_params(self, batch: OBCBatch) -> Dict[str, float]:
        """
        Обновляет параметры policy используя PPO + Imitation loss.
        
        Loss = ppo_weight * PPO_surrogate + imitation_weight * MSE(action, expert) + value_weight * MSE(value, returns)
        
        Args:
            batch: OBCBatch с данными
        
        Returns:
            Dict с метриками обновления
        """
        self.policy.train()
        
        # Prepare data
        states = batch.states.to(self.device)
        actions = batch.student_actions.to(self.device)
        expert_actions = batch.expert_actions.to(self.device)
        returns = batch.returns.to(self.device)
        advantages = batch.advantages.to(self.device)
        fixed_log_probs = batch.log_probs.to(self.device)
        
        batch_size = states.shape[0]
        ppo_losses, imitation_losses, value_losses, entropy_losses = [], [], [], []

        # PPO epochs
        for ppo_epoch in range(self.opt_num_epochs):
            # Shuffle indices
            indices = np.arange(batch_size)
            
            np.random.shuffle(indices)

            # Mini-batch updates
            n_batches = max(1, batch_size // self.batch_size)
  
            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, batch_size)
                batch_indices = indices[start_idx:end_idx]
                mini_states = states[batch_indices]
                mini_actions = actions[batch_indices]
                mini_expert = expert_actions[batch_indices]
                mini_returns = returns[batch_indices]
                mini_advantages = advantages[batch_indices]
                mini_fixed_log_probs = fixed_log_probs[batch_indices]

                # Forward pass
                pred_actions, log_std, values = self.policy(
                    mini_states,
                    batch.obs_signatures,
                    batch.action_signatures
                )

                new_log_probs = self.policy._compute_log_prob(
                    mini_actions,
                    pred_actions,
                    log_std
                )
                
                # PPO Clipped Surrogate Loss
                ratio = torch.exp(new_log_probs - mini_fixed_log_probs)
                surr1 = ratio * mini_advantages
                surr2 = torch.clamp(
                    ratio, 
                    1.0 - self.clip_epsilon, 
                    1.0 + self.clip_epsilon
                ) * mini_advantages
                ppo_loss = -torch.min(surr1, surr2).mean()
                
                # Imitation loss (MSE к эксперту)
                imitation_loss = nn.functional.mse_loss(pred_actions, mini_expert)
                
                # Value loss (MSE к GAE returns)
                value_loss = nn.functional.mse_loss(values, mini_returns)
                
                # Entropy loss (Gaussian entropy: H = 0.5 * (1 + log(2π)) + log_std)
                # Для поощрения exploration максимизируем entropy → минимизируем -entropy
                entropy = 0.5 * (1 + 1.8378770664093453) + log_std.mean()  # log(2π) ≈ 1.8379
                entropy_loss = -entropy  # Минус, т.к. хотим максимизировать entropy

                loss = 0
                if self.ppo_weight > 0:
                    loss += self.ppo_weight * ppo_loss
                if self.imitation_weight > 0:
                    loss += self.imitation_weight * imitation_loss
                if self.value_weight > 0:
                    loss += self.value_weight * value_loss
                if self.entropy_weight > 0:
                    loss += self.entropy_weight * entropy_loss
                
                # Проверка на NaN/Inf перед backward
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf detected in loss! Skipping this mini-batch. "
                                   f"ppo={ppo_loss.item():.4f}, im={imitation_loss.item():.4f}, "
                                   f"val={value_loss.item():.4f}, ent={entropy_loss.item():.4f}")
                    continue
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                
                self.optimizer.step()

                # Собираем все 4 лосса для логирования
                ppo_losses.append(ppo_loss.item())
                imitation_losses.append(imitation_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        return {
            "ppo_loss": float(np.mean(ppo_losses)) if ppo_losses else 0.0,
            "imitation_loss": float(np.mean(imitation_losses)) if imitation_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy_loss": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
        }
    
    def optimize_policy(self) -> None:
        """
        Главный цикл обучения OBC.
        """
        logger.info("Starting OBC training...")

        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            
            # Pre-epoch hook (resample motions if needed)
            if hasattr(self.expert.env, 'sample_motions') and epoch > 0:
                if epoch % self.resampling_interval == 0:
                    self.expert.env.sample_motions()
            
            # Sample trajectories
            t_sample_start = time.time()
            batch, obc_logger = self.sample(self.min_batch_size)
            t_sample = time.time() - t_sample_start

            # Save debug checkpoint BEFORE update (если включено)
            if self.debug_checkpoints:
                self.save_debug_checkpoint(batch, epoch)

            # Update parameters
            t_update_start = time.time()
            losses = self.update_params(batch)
            t_update = time.time() - t_update_start

            # Прокидываем времена и лоссы в логгер
            obc_logger.sample_time = t_sample
            obc_logger.update_time = t_update
            obc_logger.set_update_losses(**losses)

            self.num_steps += obc_logger.num_steps
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            if epoch % self.log_frequency == 0:
                self.log_train(epoch, obc_logger)
                        
            # Evaluation на валидационном сете
            if self.eval_frequency > 0 and epoch > 0 and epoch % self.eval_frequency == 0:
                eval_metrics = self.evaluate()
                
                if eval_metrics:
                    # Track best eval imitation loss (меньше = лучше)
                    if eval_metrics["eval/imitation_loss"] < self.best_eval_imitation_loss:
                        self.best_eval_imitation_loss = eval_metrics["eval/imitation_loss"]
                        self.save_checkpoint(suffix="best_im_loss")
                    
                    # Track best eval episode length (больше = лучше)
                    if eval_metrics["eval/mean_length"] > self.best_eval_episode_avg_length:
                        self.best_eval_episode_avg_length = eval_metrics["eval/mean_length"]
                        self.save_checkpoint(suffix="best_ep_length")
                    
                    # Log to WandB
                    if self.use_wandb:
                        self.wandb_logger.log_eval(epoch, eval_metrics)
            
            # Save current checkpoint
            if epoch > 0 and epoch % self.save_curr_frequency == 0:
                self.save_checkpoint(suffix="latest")
            
            # Save numbered checkpoint
            if epoch > 0 and epoch % self.save_frequency == 0:
                self.save_checkpoint(suffix=f"epoch_{epoch:05d}")

        # Final evaluation и сохранение лучших моделей
        if self.eval_frequency > 0:
            logger.info("Final evaluation...")
            eval_metrics = self.evaluate()
            
            if eval_metrics:
                # Проверяем и сохраняем лучшие модели
                if eval_metrics["eval/imitation_loss"] < self.best_eval_imitation_loss:
                    self.best_eval_imitation_loss = eval_metrics["eval/imitation_loss"]
                    self.save_checkpoint(suffix="best_im_loss")
                
                if eval_metrics["eval/mean_length"] > self.best_eval_episode_avg_length:
                    self.best_eval_episode_avg_length = eval_metrics["eval/mean_length"]
                    self.save_checkpoint(suffix="best_ep_length")
        
        # Final save
        self.save_checkpoint(suffix="latest")
        
        logger.info("OBC training completed!")
        
        # Finish wandb
        if self.use_wandb:
            self.wandb_logger.finish()
    
    def log_train(
        self,
        epoch: int,
        obc_logger: OBCLogger
    ) -> None:
        """
        Логирует метрики эпохи.

        Формирование строкового сообщения делегируем самому OBCLogger,
        чтобы формат не дублировался в двух местах.
        """
        log_str = obc_logger.get_log_str(
            epoch=epoch,
            exp_name=self.exp_name
        )
        logger.info(log_str)

        # WandB logging через WandbLogger
        if self.use_wandb:
            self.wandb_logger.log_train(
                epoch=epoch,
                obc_logger=obc_logger,
                total_steps=self.num_steps,
            )
    
    def save_checkpoint(self, suffix: str = "latest") -> None:
        """
        Сохраняет чекпоинт.
        
        Args:
            suffix: Suffix for the checkpoint file
        """
        checkpoint = {
            "epoch": self.epoch,
            "num_steps": self.num_steps,
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        
        path = os.path.join(self.output_dir, f"model_{suffix}.pth")
        torch.save(checkpoint, path)    
        logger.info(f"Checkpoint saved: {path}")

    def save_debug_checkpoint(self, batch: OBCBatch, epoch: int) -> None:
        """
        Сохраняет дебаг-чекпоинт ПЕРЕД update_params для детального анализа.
        
        Сохраняет:
        - Веса нейронной сети
        - Состояние оптимизатора
        - Батч данных (для воспроизведения update)
        - Состояние scheduler (если есть)
        
        Args:
            batch: OBCBatch с данными для обновления
            epoch: Номер эпохи
        """
        debug_dir = os.path.join(self.output_dir, "debug_checkpoints")
        os.makedirs(debug_dir, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "num_steps": self.num_steps,
            # Веса сети
            "policy": self.policy.state_dict(),
            # Оптимизатор
            "optimizer": self.optimizer.state_dict(),
            # Батч данных
            "batch": {
                "states": batch.states.cpu(),
                "student_actions": batch.student_actions.cpu(),
                "expert_actions": batch.expert_actions.cpu(),
                "rewards": batch.rewards.cpu(),
                "values": batch.values.cpu(),
                "returns": batch.returns.cpu(),
                "advantages": batch.advantages.cpu(),
                "masks": batch.masks.cpu(),
                "log_probs": batch.log_probs.cpu(),
                "obs_signatures": batch.obs_signatures,
                "action_signatures": batch.action_signatures,
            },
            # Гиперпараметры для воспроизведения
            "hyperparams": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "opt_num_epochs": self.opt_num_epochs,
                "grad_clip": self.grad_clip,
                "ppo_weight": self.ppo_weight,
                "imitation_weight": self.imitation_weight,
                "value_weight": self.value_weight,
                "entropy_weight": self.entropy_weight,
                "clip_epsilon": self.clip_epsilon,
            },
        }
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()
        
        path = os.path.join(debug_dir, f"debug_epoch_{epoch:05d}.pth")
        torch.save(checkpoint, path)
        logger.debug(f"Debug checkpoint saved: {path}")
    
    def load_checkpoint(self, epoch: int) -> None:
        """
        Загружает чекпоинт.
        
        Args:
            epoch: Номер эпохи (0 = нет загрузки, -1 = latest, >0 = конкретная эпоха)
        """
        if epoch == 0:
            logger.info("Starting from scratch (epoch=0)")
            return
        
        if epoch == -1:
            path = os.path.join(self.output_dir, "model.pth")
        else:
            path = os.path.join(self.output_dir, f"model_epoch_{epoch:05d}.pth")
        
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint["epoch"] + 1  # Resume from next epoch
        self.num_steps = checkpoint["num_steps"]
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        if self.scheduler is not None and "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}, resuming from epoch {self.epoch}")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Оценивает policy на валидационном сете движений.
        
        Использует паттерн из Kinesis AgentIM.eval_policy():
        - start_eval() -> forward_motions() -> reset() -> episode -> end_eval()
        
        Измеряет:
        - Imitation loss (MSE между действиями студента и эксперта)
        - Value loss (MSE между предсказанными values и returns)
        - Средняя длина эпизода
        - Средняя награда
        
        Returns:
            Dict с метриками
        """
        self.policy.eval()
        
        # Создаём parser для valid эксперта
        valid_parser = ObservationParser.from_env(self.valid_expert.env, self.history_len)
        
        episode_rewards = []
        episode_lengths = []
        episode_imitation_losses = []
        episode_value_losses = []
        
        for motion_id in tqdm(
            self.valid_expert.forward_motions(),
            total=self.valid_expert.num_motions,
            desc="Evaluating",
        ):

            # Reset для текущего загруженного движения
            obs, info = self.valid_expert.reset()
            valid_parser.reset(obs)
            
            episode_reward = 0.0
            episode_length = 0
            step_imitation_losses = []
            step_values = []
            step_rewards = []
            
            for t in range(10000):
                obs_ts, obs_sigs = valid_parser.get_observation(self.device)
                act_sigs = valid_parser.get_action_signatures()
                
                with torch.no_grad():
                    # Получаем mean action (deterministic) и value
                    action, _, value = self.policy.get_action(
                        obs_ts, obs_sigs, act_sigs, deterministic=True
                    )
                    
                    # Получаем действие эксперта
                    expert_action = self.valid_expert.get_expert_action(obs)
                    expert_action_t = torch.from_numpy(expert_action).float().to(self.device)
                    
                    # Imitation loss (MSE)
                    imitation_loss = ((action.squeeze(0) - expert_action_t) ** 2).mean().item()
                    step_imitation_losses.append(imitation_loss)
                    step_values.append(value.item())
                
                action_np = action.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = self.valid_expert.step(action_np)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                step_rewards.append(reward)
                
                valid_parser.update(next_obs)
                obs = next_obs
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_imitation_losses.append(np.mean(step_imitation_losses))
            
            # Value loss: сравниваем предсказанные values с cumulative rewards (простая оценка)
            cumulative_rewards = np.cumsum(step_rewards[::-1])[::-1]
            if len(step_values) == len(cumulative_rewards):
                value_loss = np.mean((np.array(step_values) - cumulative_rewards) ** 2)
                episode_value_losses.append(value_loss)
        
        metrics = {
            "eval/mean_reward": float(np.mean(episode_rewards)),
            "eval/std_reward": float(np.std(episode_rewards)),
            "eval/mean_length": float(np.mean(episode_lengths)),
            "eval/std_length": float(np.std(episode_lengths)),
            "eval/imitation_loss": float(np.mean(episode_imitation_losses)),
            "eval/value_loss": float(np.mean(episode_value_losses))
        }
        
        logger.info(
            f"Eval: reward={metrics['eval/mean_reward']:.4f}±{metrics['eval/std_reward']:.4f}, "
            f"length={metrics['eval/mean_length']:.2f}±{metrics['eval/std_length']:.2f}, "
            f"im_loss={metrics['eval/imitation_loss']:.4f}, val_loss={metrics['eval/value_loss']:.4f}"
        )
        
        return metrics
