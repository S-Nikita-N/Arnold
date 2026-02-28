"""
Arnold Trainer — универсальный трейнер для Arnold.

Поддерживает три режима обучения:
1. OBC (On-Policy Behavior Cloning):
   ppo_weight=0, imitation_weight>0
   → Чистая дистилляция из эксперта без PPO surrogate loss.

2. OBC-PPO:
   ppo_weight>0, imitation_weight>0
   → Комбинация PPO и имитации — дистилляция с reinforcement learning.

3. PPO:
   ppo_weight>0, imitation_weight=0
   → Чистый PPO без эксперта (не нужен загруженный эксперт).

Режим определяется автоматически по весам лоссов (ppo_weight, imitation_weight).

Эксперты/среды задаются списком cfg.run.experts:
  - Один элемент — обычное обучение.
  - Несколько элементов — мульти-задачное (не реализовано, бросает NotImplementedError).

Поддерживаемые типы: kinesis, myohuman.
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
from arnold.memory import OBCMemory, OBCBatch
from arnold.logger import OBCLogger
from arnold.wandb_logger import WandbLogger
from arnold.learning_utils import to_test, to_cpu, optimizer_to
from arnold.profiler import SamplingProfiler

# Игнорируем SyntaxWarning про invalid escape sequence в docstrings Kinesis
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")

os.environ["OMP_NUM_THREADS"] = "1"


logger = logging.getLogger(__name__)


def create_expert_wrapper(
    expert_entry: DictConfig,
    mode: str = "train",
    overrides: list = None,
):
    """
    Создаёт обёртку для одного эксперта/среды.

    Args:
        expert_entry: Элемент из cfg.run.experts (dict).
                      Общие поля: type, config_path, checkpoint_epoch.
                      Myohuman: simple (bool) — упрощённая модель (simpletorso).
        mode: "train" или "valid"
        overrides: Дополнительные Hydra overrides
    """
    expert_type = expert_entry.type
    expert_cfg_path = expert_entry.get("config_path", None)
    checkpoint_epoch = expert_entry.get("checkpoint_epoch", -1)

    if expert_type == "kinesis":
        from arnold.experts.kinesis_wrapper import KinesisWrapper

        return KinesisWrapper(
            cfg_path=expert_cfg_path,
            checkpoint_epoch=checkpoint_epoch,
            device="cpu",
            overrides=overrides or [],
            mode=mode,
        )

    elif expert_type == "myohuman":
        from arnold.experts.myohuman_wrapper import MyoHumanWrapper

        simple = expert_entry.get("simple", False)

        return MyoHumanWrapper(
            cfg_path=expert_cfg_path,
            checkpoint_epoch=checkpoint_epoch,
            device="cpu",
            overrides=overrides or [],
            mode=mode,
            simple=simple,
        )

    else:
        raise ValueError(
            f"Unknown expert type: '{expert_type}'. "
            f"Supported: 'kinesis', 'myohuman'"
        )


class ArnoldTrainer:
    """
    Универсальный трейнер для Arnold.

    Поддерживает режимы OBC, OBC-PPO, PPO.
    Режим определяется автоматически по весам лоссов:
    - imitation_weight > 0 → нужен эксперт (OBC или OBC-PPO)
    - imitation_weight == 0 → PPO-only (эксперт не нужен)

    Usage:
        cfg = OmegaConf.load("config.yaml")
        trainer = ArnoldTrainer(cfg, device="cuda")
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
        self.num_enc_layers = cfg.learning.num_enc_layers
        self.num_act_dec_layers = cfg.learning.num_act_dec_layers
        self.num_val_dec_layers = cfg.learning.num_val_dec_layers
        self.dropout = cfg.learning.dropout
        self.tokenizer_granularity = cfg.learning.tokenizer_granularity

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
        self.detached_value_encoder = cfg.learning.detached_value_encoder
        self.min_batch_size = cfg.learning.min_batch_size
        self.max_epochs = cfg.learning.max_epochs
        self.use_scheduler = cfg.learning.use_scheduler
        self.use_compile = cfg.learning.use_compile

        # Run (из cfg.run)
        self.num_threads = cfg.run.num_threads
        self.save_frequency = cfg.run.save_frequency
        self.save_curr_frequency = cfg.run.save_curr_frequency
        self.log_frequency = cfg.run.log_frequency
        self.output_dir = cfg.run.output_dir
        self.eval_frequency = cfg.run.eval_frequency
        self.resampling_interval = cfg.run.resampling_interval

        # Logging
        self.use_wandb = cfg.use_wandb
        self.no_log = cfg.no_log
        self.exp_name = cfg.exp_name

        # Resume
        self.checkpoint_epoch = cfg.epoch
        self.resume_checkpoint = cfg.get("resume_checkpoint", None)

        # Debug mode — сохранение чекпоинтов перед каждым update_params
        self.debug_checkpoints = getattr(cfg.learning, 'debug_checkpoints', False)

        # ==================== Определяем режим ====================
        self.use_expert = self.imitation_weight > 0
        if self.use_expert and self.ppo_weight > 0:
            self.training_mode = "obc-ppo"
        elif self.use_expert:
            self.training_mode = "obc"
        else:
            self.training_mode = "ppo"

        # State
        self.epoch = 0
        self.num_steps = 0

        # Best model tracking (на eval)
        self.best_eval_imitation_loss = float('inf')
        self.best_eval_episode_avg_length = 0.0

        # Multiprocessing Event
        self.mp_done = multiprocessing.Event()

        # Профайлер семплирования (только main process)
        self.sampling_profiler = SamplingProfiler()

        # Setup
        self.setup_expert()
        self.setup_parser()
        self.setup_policy()
        self.setup_optimizer()

        # Load checkpoint if specified
        if self.resume_checkpoint:
            self._load_from_path(
                self.resume_checkpoint,
                resume_training=not cfg.learning.transfer_mode
            )
        else:
            self.load_checkpoint(self.checkpoint_epoch)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize WandB logger
        if self.use_wandb:
            self.wandb_logger = WandbLogger(cfg)

        logger.info(f"Training mode: {self.training_mode.upper()}")
        logger.info(
            f"  ppo_weight={self.ppo_weight}, imitation_weight={self.imitation_weight}, "
            f"value_weight={self.value_weight}, entropy_weight={self.entropy_weight}"
        )

    def setup_expert(self) -> None:
        """
        Загружает среды/экспертов из списка cfg.run.experts.

        Пока поддерживается только один эксперт (len == 1).
        Если передано больше одного — бросается NotImplementedError
        (требуется кросс-задачная нормализация наград/лоссов).
        """
        experts_list = list(self.cfg.run.experts.values())
        if len(experts_list) == 0:
            raise ValueError("cfg.run.experts is empty — нужна хотя бы одна среда.")
        if len(experts_list) > 1:
            raise NotImplementedError(
                f"Передано {len(experts_list)} экспертов, но мульти-задачное обучение "
                f"ещё не реализовано (нужна кросс-задачная нормализация наград/лоссов). "
                f"Пока поддерживается только один эксперт."
            )

        expert_entry = experts_list[0]

        logger.info(f"Setting up environment (type: {expert_entry.type}, mode: {self.training_mode})...")

        self.expert = create_expert_wrapper(expert_entry, mode="train")

        if self.use_expert:
            if hasattr(self.expert, 'has_expert') and not self.expert.has_expert:
                logger.warning(
                    "imitation_weight > 0, but expert policy not loaded! "
                    "Imitation loss will fail. Set imitation_weight=0 for PPO-only."
                )

        logger.info(f"Environment loaded. Obs dim: {self.expert.obs_dim}, Action dim: {self.expert.action_dim}")

        # Valid expert для evaluation
        self.valid_expert = None
        if self.eval_frequency > 0:
            logger.info("Loading validation environment...")
            self.valid_expert = create_expert_wrapper(expert_entry, mode="valid")

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

        # Body groups для tokenization
        groups = self.parser.get_body_groups(self.tokenizer_granularity)
        logger.info(
            f"Tokenizer [{self.tokenizer_granularity}]: "
            f"{len(groups)} tokens (from {self.parser.n_obs_elements} flat obs)"
        )

        # Policy
        self.policy = TransformerPolicy(
            vocab=self.vocab,
            groups=groups,
            history_len=self.history_len,
            embed_dim=self.embed_dim,
            ff_dim=self.ff_dim,
            num_heads=self.num_heads,
            num_enc_layers=self.num_enc_layers,
            num_act_dec_layers=self.num_act_dec_layers,
            num_val_dec_layers=self.num_val_dec_layers,
            dropout=self.dropout,
            detached_value_encoder=self.detached_value_encoder,
        )
        self.policy.to(self.device)

        if self.use_compile:
            self.policy = torch.compile(self.policy, mode="reduce-overhead")
            logger.info("torch.compile applied (mode=reduce-overhead)")

        logger.info(f"Arnold policy created. Parameters: {sum(p.numel() for p in self.policy.parameters()):,}")

    def setup_optimizer(self) -> None:
        """Создаёт оптимизатор."""
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_epochs,
                eta_min=self.learning_rate * 0.1,
            )
        else:
            self.scheduler = None

    def seed_worker(self, pid: int) -> None:
        """Устанавливает seed для worker процесса."""
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

        Поддерживает PPO-only режим: если self.use_expert == False,
        пропускает вычисление expert_action и записывает нули.
        """
        self.seed_worker(pid)

        worker_parser = ObservationParser.from_env(self.expert.env, self.history_len)

        memory = OBCMemory()
        obc_logger = OBCLogger()

        # Профайлер только в main process (pid==0) и только эпоха 0
        profiler: Optional[SamplingProfiler] = None
        if pid == 0:
            if self.epoch == 0:
                profiler = self.sampling_profiler
                profiler.reset()
                self.policy.enable_profiling(profiler)
            pbar = tqdm(total=min_batch_size, desc="Sampling", unit="step")

        # Для PPO-only: заготовка нулевого expert_action
        if not self.use_expert:
            zero_expert = torch.zeros(self.expert.action_dim, dtype=torch.float32)

        try:
            while obc_logger.num_steps < min_batch_size:
                obs, info = self.expert.reset()
                worker_parser.reset(obs)

                for t in range(10000):
                    # --- parser.get_observation ---
                    if profiler: profiler.tick("parser.get_obs")
                    obs_ts, obs_sigs = worker_parser.get_observation(torch.device("cpu"))
                    act_sigs = worker_parser.action_signatures
                    if profiler: profiler.tock("parser.get_obs")

                    # --- policy.get_action (внутри — sub-компоненты через хуки в forward) ---
                    if profiler: profiler.tick("policy.forward")
                    with torch.no_grad():
                        student_action, log_prob, value = self.policy.get_action(
                            obs_ts, obs_sigs, act_sigs, deterministic=False
                        )
                    if profiler: profiler.tock("policy.forward")

                    # --- expert.get_expert_action ---
                    if self.use_expert:
                        if profiler: profiler.tick("expert.get_action")
                        expert_action = self.expert.get_expert_action(obs)
                        expert_action_t = torch.from_numpy(expert_action).float()
                        if profiler: profiler.tock("expert.get_action")
                    else:
                        expert_action_t = zero_expert

                    # --- env.step ---
                    student_action_np = student_action.squeeze(0).cpu().numpy()
                    if profiler: profiler.tick("env.step")
                    next_obs, reward, terminated, truncated, info = self.expert.step(student_action_np)
                    if profiler: profiler.tock("env.step")
                    done = terminated or truncated

                    # --- memory.append ---
                    if profiler: profiler.tick("memory.append")
                    memory.states.append(obs_ts.squeeze(0).cpu())
                    memory.obs_signatures.append(obs_sigs)
                    memory.action_signatures.append(act_sigs)
                    memory.student_actions.append(student_action.squeeze(0).cpu())
                    memory.expert_actions.append(expert_action_t.cpu())
                    memory.rewards.append(reward)
                    memory.values.append(value.squeeze(0).cpu())
                    memory.masks.append(0.0 if done else 1.0)
                    memory.log_probs.append(log_prob.squeeze(0).cpu())
                    if profiler: profiler.tock("memory.append")

                    obc_logger.step(
                        reward=reward,
                        info=info if isinstance(info, dict) else None,
                    )

                    if pid == 0:
                        pbar.update(1)

                    if done:
                        break

                    # --- parser.update ---
                    if profiler: profiler.tick("parser.update")
                    worker_parser.update(next_obs)
                    if profiler: profiler.tock("parser.update")
                    obs = next_obs

                obc_logger.end_episode()

        except Exception as e:
            import traceback
            print(f"Worker {pid} failed: {e}")
            traceback.print_exc()

        finally:
            if pid == 0:
                pbar.close()
                if profiler is not None:
                    self.policy.disable_profiling()

            if queue is not None:
                queue.put([pid, memory.to_transfer_dict(), obc_logger])
                mp_done.wait()
            else:
                return memory, obc_logger

    def sample(self, min_batch_size: int) -> Tuple[OBCBatch, OBCLogger]:
        """
        Собирает траектории.
        Поддерживает многопроцессную сборку.
        """
        t_start = time.time()

        # Сбрасываем Event перед новой эпохой
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
                    worker_args = (i + 1, queue, self.mp_done, thread_batch_size, self.cfg.run.experts)
                    worker = multiprocessing.Process(
                        target=self.sample_worker,
                        args=worker_args
                    )
                    worker.start()

                # Main process samples (pid = 0)
                memories[0], loggers[0] = self.sample_worker(
                    0, None, self.mp_done, thread_batch_size, self.cfg.run.experts
                )

                # Collect from workers
                for i in range(self.num_threads - 1):
                    pid, worker_transfer, worker_logger = queue.get()
                    memories[pid] = OBCMemory.from_transfer_dict(worker_transfer)
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
        """Обёртка над OBCMemory.to_batch."""
        return memory.to_batch(
            gamma=self.gamma,
            tau=self.tau,
            device=None,
        )

    def update_params(self, batch: OBCBatch) -> Dict[str, float]:
        """
        Обновляет параметры policy.

        Loss = ppo_weight * PPO + imitation_weight * MSE(action, expert)
             + value_weight * MSE(value, returns) + entropy_weight * (-entropy)

        Лоссы с нулевым весом не вычисляются (экономим compute).
        """
        self.policy.train()

        # Prepare data
        states = batch.states.to(self.device)
        actions = batch.student_actions.to(self.device)
        returns = batch.returns.to(self.device)
        advantages = batch.advantages.to(self.device)
        fixed_log_probs = batch.log_probs.to(self.device)
        old_values = batch.values.to(self.device)

        if self.use_expert:
            expert_actions = batch.expert_actions.to(self.device)

        batch_size = states.shape[0]
        ppo_losses, imitation_losses, value_losses, entropy_losses = [], [], [], []

        # PPO diagnostics accumulators
        all_ratios = []
        all_clip_fracs = []
        all_approx_kls = []
        all_grad_norms = []
        kl_early_stop_epoch = -1

        # target_kl = 0.05

        n_batches = max(1, batch_size // self.batch_size)
        total_updates = self.opt_num_epochs * n_batches
        pbar = tqdm(total=total_updates, desc="Update", unit="batch")

        for ppo_epoch in range(self.opt_num_epochs):
            epoch_kls = []
            indices = np.arange(batch_size)
            np.random.shuffle(indices)

            for i in range(n_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, batch_size)
                batch_indices = indices[start_idx:end_idx]
                mini_states = states[batch_indices]
                mini_actions = actions[batch_indices]
                mini_returns = returns[batch_indices]
                mini_advantages = advantages[batch_indices]
                mini_fixed_log_probs = fixed_log_probs[batch_indices]
                mini_old_values = old_values[batch_indices]

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

                loss = 0

                if self.ppo_weight > 0:
                    log_ratio = new_log_probs - mini_fixed_log_probs
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * mini_advantages
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon
                    ) * mini_advantages
                    ppo_loss = -torch.min(surr1, surr2).mean()
                    loss = loss + self.ppo_weight * ppo_loss
                    ppo_losses.append(ppo_loss.item())

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                        all_ratios.append(ratio.mean().item())
                        all_clip_fracs.append(clip_frac)
                        all_approx_kls.append(approx_kl)
                        epoch_kls.append(approx_kl)

                if self.use_expert and self.imitation_weight > 0:
                    mini_expert = expert_actions[batch_indices]
                    imitation_loss = nn.functional.mse_loss(pred_actions, mini_expert)
                    loss = loss + self.imitation_weight * imitation_loss
                    imitation_losses.append(imitation_loss.item())

                if self.value_weight > 0:
                    vf_loss_unclipped = (values - mini_returns) ** 2
                    v_clipped = mini_old_values + torch.clamp(
                        values - mini_old_values,
                        -self.clip_epsilon,
                        self.clip_epsilon,
                    )
                    vf_loss_clipped = (v_clipped - mini_returns) ** 2
                    value_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()
                    loss = loss + self.value_weight * value_loss
                    value_losses.append(value_loss.item())

                if self.entropy_weight > 0:
                    entropy = 0.5 * (1 + 1.8378770664093453) + log_std.mean()
                    entropy_loss = -entropy
                    loss = loss + self.entropy_weight * entropy_loss
                    entropy_losses.append(entropy_loss.item())

                self.optimizer.zero_grad()
                loss.backward()

                if self.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                    all_grad_norms.append(grad_norm.item())
                else:
                    total_norm = sum(p.grad.norm().item() ** 2 for p in self.policy.parameters() if p.grad is not None) ** 0.5
                    all_grad_norms.append(total_norm)

                self.optimizer.step()
                pbar.update(1)

            # # KL early stopping: проверяем после каждой PPO epoch
            # if epoch_kls and np.mean(epoch_kls) > target_kl:
            #     kl_early_stop_epoch = ppo_epoch
            #     remaining = (self.opt_num_epochs - ppo_epoch - 1) * n_batches
            #     pbar.update(remaining)
            #     break

        pbar.close()

        # Post-update: прогоним один батч через обновлённую policy для диагностики выходов
        with torch.no_grad():
            diag_idx = np.arange(min(512, batch_size))
            diag_actions, diag_log_std, diag_values = self.policy(
                states[diag_idx], batch.obs_signatures, batch.action_signatures
            )
            act_mean = diag_actions.mean().item()
            act_std = diag_actions.std().item()
            act_abs_mean = diag_actions.abs().mean().item()
            act_min = diag_actions.min().item()
            act_max = diag_actions.max().item()
            logstd_mean = diag_log_std.mean().item() if diag_log_std is not None else 0.0
            logstd_min = diag_log_std.min().item() if diag_log_std is not None else 0.0
            logstd_max = diag_log_std.max().item() if diag_log_std is not None else 0.0
            val_post_mean = diag_values.mean().item() if diag_values is not None else 0.0
            val_post_std = diag_values.std().item() if diag_values is not None else 0.0

        # Sampling-phase diagnostics (pre-update data)
        with torch.no_grad():
            adv_mean = advantages.mean().item()
            adv_std = advantages.std().item()
            ret_mean = returns.mean().item()
            ret_std = returns.std().item()
            val_mean = old_values.mean().item()
            val_std = old_values.std().item()
            ret_var = returns.var().item()
            explained_var = 1 - (returns - old_values).var().item() / (ret_var + 1e-8)

        sigma_global = self.policy.log_sigma_global.item()

        return {
            "ppo_loss": float(np.mean(ppo_losses)) if ppo_losses else 0.0,
            "imitation_loss": float(np.mean(imitation_losses)) if imitation_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy_loss": float(np.mean(entropy_losses)) if entropy_losses else 0.0,
            "approx_kl": float(np.mean(all_approx_kls)) if all_approx_kls else 0.0,
            "clip_frac": float(np.mean(all_clip_fracs)) if all_clip_fracs else 0.0,
            "ratio_mean": float(np.mean(all_ratios)) if all_ratios else 1.0,
            "grad_norm": float(np.mean(all_grad_norms)) if all_grad_norms else 0.0,
            "explained_var": explained_var,
            "adv_mean": adv_mean,
            "adv_std": adv_std,
            "ret_mean": ret_mean,
            "ret_std": ret_std,
            "val_mean": val_mean,
            "val_std": val_std,
            "sigma_global": sigma_global,
            "act_mean": act_mean,
            "act_std": act_std,
            "act_abs_mean": act_abs_mean,
            "act_min": act_min,
            "act_max": act_max,
            "logstd_mean": logstd_mean,
            "logstd_min": logstd_min,
            "logstd_max": logstd_max,
            "val_post_mean": val_post_mean,
            "val_post_std": val_post_std,
            "kl_early_stop_epoch": kl_early_stop_epoch,
            "ppo_epochs_actual": (kl_early_stop_epoch + 1) if kl_early_stop_epoch >= 0 else self.opt_num_epochs,
        }

    def optimize_policy(self) -> None:
        """Главный цикл обучения."""
        logger.info(f"Starting {self.training_mode.upper()} training...")

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

            # Update parameters
            t_update_start = time.time()
            losses = self.update_params(batch)
            t_update = time.time() - t_update_start

            obc_logger.sample_time = t_sample
            obc_logger.update_time = t_update
            obc_logger.set_update_losses(
                ppo_loss=losses["ppo_loss"],
                imitation_loss=losses["imitation_loss"],
                value_loss=losses["value_loss"],
                entropy_loss=losses["entropy_loss"],
            )
            self._last_update_diagnostics = losses

            self.num_steps += obc_logger.num_steps

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            if epoch % self.log_frequency == 0:
                self.log_train(epoch, obc_logger)

            # Evaluation
            if self.eval_frequency > 0 and epoch > 0 and epoch % self.eval_frequency == 0:
                eval_metrics = self.evaluate()

                if eval_metrics:
                    if self.use_expert and eval_metrics.get("eval/imitation_loss", float('inf')) < self.best_eval_imitation_loss:
                        self.best_eval_imitation_loss = eval_metrics["eval/imitation_loss"]
                        self.save_checkpoint(suffix="best_im_loss")

                    if eval_metrics["eval/mean_length"] > self.best_eval_episode_avg_length:
                        self.best_eval_episode_avg_length = eval_metrics["eval/mean_length"]
                        self.save_checkpoint(suffix="best_ep_length")

                    if self.use_wandb:
                        self.wandb_logger.log_eval(epoch, eval_metrics)

            # Save current checkpoint
            if epoch > 0 and epoch % self.save_curr_frequency == 0:
                self.save_checkpoint(suffix="latest")

            # Save numbered checkpoint
            if epoch > 0 and epoch % self.save_frequency == 0:
                self.save_checkpoint(suffix=f"epoch_{epoch:05d}")

        # Final evaluation
        if self.eval_frequency > 0:
            logger.info("Final evaluation...")
            eval_metrics = self.evaluate()

            if eval_metrics:
                if self.use_expert and eval_metrics.get("eval/imitation_loss", float('inf')) < self.best_eval_imitation_loss:
                    self.best_eval_imitation_loss = eval_metrics["eval/imitation_loss"]
                    self.save_checkpoint(suffix="best_im_loss")

                if eval_metrics["eval/mean_length"] > self.best_eval_episode_avg_length:
                    self.best_eval_episode_avg_length = eval_metrics["eval/mean_length"]
                    self.save_checkpoint(suffix="best_ep_length")

        # Final save
        self.save_checkpoint(suffix="latest")

        logger.info(f"{self.training_mode.upper()} training completed!")

        if self.use_wandb:
            self.wandb_logger.finish()

    def log_train(self, epoch: int, obc_logger: OBCLogger) -> None:
        """Логирует метрики эпохи."""
        log_str = obc_logger.get_log_str(epoch=epoch, exp_name=self.exp_name)
        logger.info(log_str)

        # Sampling profiler — только эпоха 0
        if epoch == 0:
            logger.info(
                f"Sampling profile (epoch {epoch}, main process):\n"
                + self.sampling_profiler.report()
            )

        # PPO diagnostics — каждую эпоху
        d = getattr(self, '_last_update_diagnostics', {})
        if d:
            ppo_ep = d.get('ppo_epochs_actual', '?')
            kl_stop = d.get('kl_early_stop_epoch', -1)
            kl_note = f" (KL stop @ ep {kl_stop})" if kl_stop >= 0 else ""
            logger.info(
                f"PPO diag: "
                f"approx_kl={d.get('approx_kl', 0):.4f}  "
                f"clip_frac={d.get('clip_frac', 0):.3f}  "
                f"ratio={d.get('ratio_mean', 1):.4f}  "
                f"grad_norm={d.get('grad_norm', 0):.3f}  "
                f"σ_global={d.get('sigma_global', 0):.4f}  "
                f"expl_var={d.get('explained_var', 0):.3f}  "
                f"ppo_epochs={ppo_ep}/{self.opt_num_epochs}{kl_note}  "
                f"adv={d.get('adv_mean', 0):.4f}±{d.get('adv_std', 0):.4f}  "
                f"ret={d.get('ret_mean', 0):.3f}±{d.get('ret_std', 0):.3f}  "
                f"val={d.get('val_mean', 0):.3f}±{d.get('val_std', 0):.3f}"
            )
            logger.info(
                f"Policy out: "
                f"act_mean={d.get('act_mean', 0):.4f}  "
                f"act_std={d.get('act_std', 0):.4f}  "
                f"|act|={d.get('act_abs_mean', 0):.4f}  "
                f"act_range=[{d.get('act_min', 0):.3f}, {d.get('act_max', 0):.3f}]  "
                f"logstd={d.get('logstd_mean', 0):.3f} [{d.get('logstd_min', 0):.3f}, {d.get('logstd_max', 0):.3f}]  "
                f"val_post={d.get('val_post_mean', 0):.3f}±{d.get('val_post_std', 0):.3f}  "
            )

        if self.use_wandb:
            self.wandb_logger.log_train(
                epoch=epoch,
                obc_logger=obc_logger,
                total_steps=self.num_steps,
            )

    def save_checkpoint(self, suffix: str = "latest") -> None:
        """Сохраняет чекпоинт."""
        checkpoint = {
            "epoch": self.epoch,
            "num_steps": self.num_steps,
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_mode": self.training_mode,
        }
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        path = os.path.join(self.output_dir, f"model_{suffix}.pth")
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, epoch: int) -> None:
        """
        Загружает чекпоинт из output_dir по номеру эпохи.

        Args:
            epoch: 0 = нет загрузки, -1 = latest, >0 = конкретная эпоха
        """
        if epoch == 0:
            logger.info("Starting from scratch (epoch=0)")
            return

        if epoch == -1:
            path = os.path.join(self.output_dir, "model.pth")
        else:
            path = os.path.join(self.output_dir, f"model_epoch_{epoch:05d}.pth")

        if not os.path.exists(path):
            latest_path = os.path.join(self.output_dir, "model_latest.pth")
            if epoch == -1 and os.path.exists(latest_path):
                path = latest_path
            else:
                logger.warning(f"Checkpoint not found: {path}")
                return

        self._load_from_path(path, resume_training=True)

    def _load_from_path(self, path: str, resume_training: bool = True) -> None:
        """
        Внутренний метод загрузки чекпоинта.

        Args:
            path: Путь к .pth файлу
            restore_optimizer: Восстановить ли состояние оптимизатора и эпоху.
                               False — для переноса между средами (только веса policy).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Загружаем policy с partial match (поддержка перехода между средами)
        try:
            self.policy.load_state_dict(checkpoint["policy"])
        except RuntimeError as e:
            logger.warning(f"Strict load failed. Exception: {e}")
            logger.info("Filtering checkpoint for size mismatches (cross-environment transfer)...")
            
            model_state = self.policy.state_dict()
            ckpt_state = checkpoint["policy"]
            filtered_state = {}
            
            for k, v in ckpt_state.items():
                if (
                    k in model_state and
                    hasattr(model_state[k], 'shape') and 
                    hasattr(v, 'shape') and 
                    model_state[k].shape != v.shape
                ):
                    logger.info(
                        f"  Skipping '{k}' due to size mismatch: "
                        f"checkpoint={v.shape} vs current={model_state[k].shape}"
                    )
                    continue
                filtered_state[k] = v
                
            self.policy.load_state_dict(filtered_state, strict=False)

        if resume_training:
            self.epoch = checkpoint["epoch"] + 1
            self.num_steps = checkpoint["num_steps"]
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler is not None and "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            # Transfer learning: сбрасываем эпоху, оптимизатор оставляем свежим
            self.epoch = 0
            self.num_steps = 0

            logger.info("Transfer mode: resetting observation normalizer statistics")
            self.policy.obs_normalizer.stats.clear()

            logger.info("Transfer mode: reinitializing value decoder and value head")
            for module in [self.policy.value_decoder, self.policy.value_head]:
                for param in module.parameters():
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    elif param.dim() == 1:
                        nn.init.zeros_(param)
            nn.init.normal_(self.policy.value_query.data)

        prev_mode = checkpoint.get("training_mode", "unknown")
        src_epoch = checkpoint.get("epoch", "?")
        logger.info(
            f"Loaded checkpoint from {path} "
            f"(epoch={src_epoch}, mode={prev_mode}), "
            f"resuming in mode {self.training_mode} from epoch {self.epoch}"
        )

    def evaluate(self) -> Dict[str, float]:
        """
        Оценивает policy на валидационном сете движений.

        Метрики:
        - Imitation loss (если есть эксперт)
        - Value loss
        - Средняя длина эпизода
        - Средняя награда
        """
        if self.valid_expert is None:
            logger.warning("No validation expert loaded, skipping evaluation.")
            return {}

        self.policy.eval()

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
            obs, info = self.valid_expert.reset()
            valid_parser.reset(obs)

            episode_reward = 0.0
            episode_length = 0
            step_imitation_losses = []
            step_values = []
            step_rewards = []

            for t in range(10000):
                obs_ts, obs_sigs = valid_parser.get_observation(self.device)
                act_sigs = valid_parser.action_signatures

                with torch.no_grad():
                    action, _, value = self.policy.get_action(
                        obs_ts, obs_sigs, act_sigs, deterministic=True
                    )

                    # Imitation loss (только если есть эксперт)
                    if self.use_expert:
                        expert_action = self.valid_expert.get_expert_action(obs)
                        expert_action_t = torch.from_numpy(expert_action).float().to(self.device)
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

            if step_imitation_losses:
                episode_imitation_losses.append(np.mean(step_imitation_losses))

            if step_values and step_rewards:
                discounted_returns = np.zeros(len(step_rewards), dtype=np.float64)
                running_return = 0.0
                for ri in reversed(range(len(step_rewards))):
                    running_return = step_rewards[ri] + self.gamma * running_return
                    discounted_returns[ri] = running_return
                value_loss = np.mean((np.array(step_values) - discounted_returns) ** 2)
                episode_value_losses.append(value_loss)

        metrics = {
            "eval/mean_reward": float(np.mean(episode_rewards)),
            "eval/std_reward": float(np.std(episode_rewards)),
            "eval/mean_length": float(np.mean(episode_lengths)),
            "eval/std_length": float(np.std(episode_lengths)),
            "eval/value_loss": float(np.mean(episode_value_losses)) if episode_value_losses else 0.0,
        }

        if episode_imitation_losses:
            metrics["eval/imitation_loss"] = float(np.mean(episode_imitation_losses))

        logger.info(
            f"Eval: reward={metrics['eval/mean_reward']:.4f}±{metrics['eval/std_reward']:.4f}, "
            f"length={metrics['eval/mean_length']:.2f}±{metrics['eval/std_length']:.2f}"
            + (f", im_loss={metrics.get('eval/imitation_loss', 0):.4f}" if self.use_expert else "")
            + f", val_loss={metrics['eval/value_loss']:.4f}"
        )

        return metrics
