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

Multi-expert: каждый эксперт автономен — свои loss-веса, память, логгер, треды.
Модель (TransformerPolicy) общая. Градиенты суммируются через interleaved mini-batches.
"""

import os
import math
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import multiprocessing as mp
import warnings

fork_ctx = mp.get_context('fork')

from dataclasses import dataclass
from omegaconf import DictConfig
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from arnold.torch_model.transformer_policy import TransformerPolicy
from arnold.torch_model.sensorimotor_vocabulary import SensorimotorVocabulary
from arnold.observation_parser import ObservationParser, BodyGroup
from arnold.memory import OBCMemory, OBCBatch
from arnold.logger import OBCLogger
from arnold.wandb_logger import WandbLogger
from arnold.learning_utils import to_test, to_cpu, optimizer_to
from arnold.profiler import SamplingProfiler

warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")

os.environ["OMP_NUM_THREADS"] = "1"


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  ExpertContext — всё, что нужно знать про одного эксперта
# ---------------------------------------------------------------------------

@dataclass
class ExpertContext:
    """Контекст одного эксперта: обёртка, парсер, loss-веса, parallelism."""

    name: str
    wrapper: Any                    # KinesisWrapper | MyoHumanWrapper
    parser: ObservationParser
    groups: List[BodyGroup]

    # Per-expert loss weights
    ppo_weight: float = 0.0
    imitation_weight: float = 0.0
    value_weight: float = 0.0
    entropy_weight: float = 0.0

    # Per-expert parallelism
    num_threads: int = 1
    min_batch_size: int = 10240

    # Per-expert loss scale (multiplier applied to all losses)
    loss_scale: float = 1.0

    @property
    def use_expert(self) -> bool:
        return self.imitation_weight > 0

    @property
    def training_mode(self) -> str:
        if self.ppo_weight > 0 and self.imitation_weight > 0:
            return "obc-ppo"
        elif self.imitation_weight > 0:
            return "obc"
        else:
            return "ppo"


# ---------------------------------------------------------------------------
#  Фабрика expert wrappers
# ---------------------------------------------------------------------------

def create_expert_wrapper(
    expert_entry: DictConfig,
    mode: str = "train",
    overrides: list = None,
):
    """
    Создаёт обёртку для одного эксперта/среды.

    Args:
        expert_entry: Элемент из cfg.run.experts (dict).
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
            f"Unknown expert type: '{expert_type}'. Supported: 'kinesis', 'myohuman'"
        )


class ArnoldTrainer:
    """
    Универсальный трейнер для Arnold.

    Multi-expert: каждый эксперт получает свой ExpertContext с loss-весами,
    памятью, логгером, тредами. Модель (TransformerPolicy) общая, projectors
    шарятся через BodyTokenizer. Градиенты от разных экспертов суммируются
    через interleaved mini-batch updates.
    """

    def __init__(
        self,
        cfg: DictConfig,
        dtype: torch.dtype = torch.float32,
        device: str = None,
    ):
        self.cfg = cfg
        self.dtype = dtype
        self.device = torch.device(device if device else cfg.device)

        # Architecture
        self.history_len = cfg.learning.history_len
        self.embed_dim = cfg.learning.embed_dim
        self.ff_dim = cfg.learning.ff_dim
        self.num_heads = cfg.learning.num_heads
        self.num_enc_layers = cfg.learning.num_enc_layers
        self.num_act_dec_layers = cfg.learning.num_act_dec_layers
        self.num_val_dec_layers = cfg.learning.num_val_dec_layers
        self.dropout = cfg.learning.dropout
        self.tokenizer_granularity = cfg.learning.tokenizer_granularity

        # Training (global defaults, per-expert overrides possible)
        self.batch_size = cfg.learning.batch_size
        self.learning_rate = cfg.learning.learning_rate
        self.weight_decay = cfg.learning.weight_decay
        self.gamma = cfg.learning.gamma
        self.tau = cfg.learning.tau
        self.clip_epsilon = cfg.learning.clip_epsilon
        self.opt_num_epochs = cfg.learning.opt_num_epochs
        self.grad_clip = cfg.learning.grad_clip
        self.detached_value_encoder = cfg.learning.detached_value_encoder
        self.max_epochs = cfg.learning.max_epochs
        self.use_scheduler = cfg.learning.use_scheduler
        self.use_compile = cfg.learning.use_compile

        # Run
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

        # Debug
        self.debug_checkpoints = getattr(cfg.learning, 'debug_checkpoints', False)

        # State
        self.epoch = 0
        self.num_steps = 0

        # Best model tracking
        self.best_eval_episode_avg_length: Dict[str, float] = {}
        self.best_eval_imitation_loss: Dict[str, float] = {}

        # Multiprocessing Event
        self.mp_done = fork_ctx.Event()

        # Per-expert profiler reports (заполняется на epoch 0)
        self.profiler_reports: Dict[str, SamplingProfiler] = {}

        # ==================== Setup ====================
        self.setup_experts()
        self.setup_policy()
        self.setup_optimizer()

        # Load checkpoint
        if self.resume_checkpoint:
            self.load_from_path(
                self.resume_checkpoint,
                resume_training=not cfg.learning.transfer_mode,
            )
        else:
            self.load_checkpoint(self.checkpoint_epoch)

        os.makedirs(self.output_dir, exist_ok=True)

        if self.use_wandb:
            self.wandb_logger = WandbLogger(cfg)

        self.log_training_config()

    # ------------------------------------------------------------------
    #  Setup
    # ------------------------------------------------------------------

    def setup_experts(self) -> None:
        """Загружает экспертов и создаёт ExpertContext для каждого."""
        experts_cfg = self.cfg.run.experts
        if not experts_cfg:
            raise ValueError("cfg.run.experts is empty — нужна хотя бы одна среда.")

        self.experts: Dict[str, ExpertContext] = {}
        self.valid_experts: Dict[str, Any] = {}

        for name, entry in experts_cfg.items():
            logger.info(self._section_header(
                f"Expert: {name!r} (type: {entry.type})"
            ))

            wrapper = create_expert_wrapper(entry, mode="train")

            parser = ObservationParser.from_env(wrapper.env, history_len=self.history_len)
            groups = parser.get_body_groups(self.tokenizer_granularity)

            learning_cfg = entry.get("learning", {})
            ppo_w = learning_cfg.get("ppo_weight")
            im_w = learning_cfg.get("imitation_weight")
            val_w = learning_cfg.get("value_weight")
            ent_w = learning_cfg.get("entropy_weight")
            loss_scale = learning_cfg.get("loss_scale", 1)
            mbs = learning_cfg.get("min_batch_size")
            n_threads = entry.get("num_threads", 1)

            ctx = ExpertContext(
                name=name,
                wrapper=wrapper,
                parser=parser,
                groups=groups,
                ppo_weight=ppo_w,
                imitation_weight=im_w,
                value_weight=val_w,
                entropy_weight=ent_w,
                num_threads=n_threads,
                min_batch_size=mbs,
                loss_scale=loss_scale,
            )
            self.experts[name] = ctx

            if ctx.use_expert and hasattr(wrapper, 'has_expert') and not wrapper.has_expert:
                logger.warning(
                    f"Expert '{name}': imitation_weight > 0 but expert policy not loaded!"
                )

            logger.info(
                f"  mode={ctx.training_mode}  "
                f"obs_dim={wrapper.obs_dim}  act_dim={wrapper.action_dim}"
            )
            logger.info(
                f"  threads={n_threads}  min_batch={mbs}  "
                f"ppo={ppo_w}  im={im_w}  val={val_w}  ent={ent_w}  scale={loss_scale}"
            )
            logger.info(
                f"  parser: {parser.n_obs_elements} obs elements, "
                f"{len(groups)} tokens [{self.tokenizer_granularity}]"
            )

            if self.eval_frequency > 0:
                logger.info("  validation env: loading...")
                valid_wrapper = create_expert_wrapper(entry, mode="valid")
                self.valid_experts[name] = valid_wrapper
                logger.info("  validation env: ready")

            self.best_eval_episode_avg_length[name] = 0.0
            self.best_eval_imitation_loss[name] = float('inf')

    def setup_policy(self) -> None:
        """Создаёт TransformerPolicy с merged группами всех экспертов."""
        logger.info("Setting up Arnold policy...")

        self.vocab = SensorimotorVocabulary(embed_dim=self.embed_dim)

        groups = {name: ctx.groups for name, ctx in self.experts.items()}

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

        logger.info(
            f"Arnold policy created. Parameters: {sum(p.numel() for p in self.policy.parameters()):,}"
        )

    def setup_optimizer(self) -> None:
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

    # ------------------------------------------------------------------
    #  Sampling
    # ------------------------------------------------------------------

    def seed_worker(self, pid: int) -> None:
        if pid > 0:
            seed = random.randint(0, 5000) * pid + self.epoch
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def sample_worker(
        self,
        pid: int,
        queue: mp.Queue,
        mp_done: mp.Event,
        min_batch_size: int,
        expert_name: str,
        step_counter: Optional[mp.Value] = None,
        enable_profiler: bool = False,
    ) -> None:
        """
        Background worker для сбора траекторий ОДНОГО эксперта.

        Результаты отправляются через queue. step_counter (shared) атомарно
        инкрементируется на каждом шаге — main process использует его для tqdm.
        enable_profiler=True — worker создаёт SamplingProfiler и отправляет
        его данные через queue (только один worker на эксперта, epoch 0).
        """
        self.seed_worker(pid)

        ctx = self.experts[expert_name]
        wrapper = ctx.wrapper
        worker_parser = ObservationParser.from_env(wrapper.env, self.history_len)

        memory = OBCMemory()
        obc_logger = OBCLogger()

        profiler = SamplingProfiler() if enable_profiler else None
        if profiler is not None:
            self.policy.enable_profiling(profiler)

        if not ctx.use_expert:
            zero_expert = torch.zeros(wrapper.action_dim, dtype=torch.float32)

        try:
            while obc_logger.num_steps < min_batch_size:
                obs, info = wrapper.reset()
                worker_parser.reset(obs)

                for t in range(10000):
                    if profiler: profiler.tick("parser.get_obs")
                    obs_ts, obs_sigs = worker_parser.get_observation(torch.device("cpu"))
                    act_sigs = worker_parser.action_signatures
                    if profiler: profiler.tock("parser.get_obs")

                    if profiler: profiler.tick("policy.forward")
                    with torch.no_grad():
                        student_action, log_prob, value = self.policy.get_action(
                            obs_ts, obs_sigs, act_sigs,
                            expert_name=expert_name,
                            deterministic=False,
                        )
                    if profiler: profiler.tock("policy.forward")

                    if ctx.use_expert:
                        if profiler: profiler.tick("expert.get_action")
                        expert_action = wrapper.get_expert_action(obs)
                        expert_action_t = torch.from_numpy(expert_action).float()
                        if profiler: profiler.tock("expert.get_action")
                    else:
                        expert_action_t = zero_expert

                    student_action_np = student_action.squeeze(0).cpu().numpy()
                    if profiler: profiler.tick("env.step")
                    next_obs, reward, terminated, truncated, info = wrapper.step(student_action_np)
                    if profiler: profiler.tock("env.step")
                    done = terminated or truncated

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

                    if step_counter is not None:
                        with step_counter.get_lock():
                            step_counter.value += 1

                    if done:
                        break

                    if profiler: profiler.tick("parser.update")
                    worker_parser.update(next_obs)
                    if profiler: profiler.tock("parser.update")
                    obs = next_obs

                obc_logger.end_episode()

        except Exception as e:
            import traceback
            print(f"Worker {pid} [{expert_name}] failed: {e}")
            traceback.print_exc()

        finally:
            if profiler is not None:
                self.policy.disable_profiling()

            profiler_data = profiler.to_dict() if profiler is not None else None
            queue.put([pid, expert_name, memory.to_transfer_dict(), obc_logger, profiler_data])
            mp_done.wait()

    def sample(self) -> Tuple[Dict[str, OBCBatch], Dict[str, OBCLogger]]:
        """
        Собирает траектории от ВСЕХ экспертов полностью параллельно.

        ВСЕ workers запускаются как background processes (num_threads на каждого
        эксперта). Main process только мониторит shared counters и обновляет
        N tqdm progress bars одновременно — без потери производительности.

        На epoch 0 первый worker каждого эксперта включает SamplingProfiler
        и передаёт результаты обратно — per-expert profiling.

        Returns:
            expert_batches: Dict[expert_name -> OBCBatch]
            expert_loggers: Dict[expert_name -> OBCLogger]
        """
        t_start = time.time()
        self.mp_done.clear()
        to_test(self.policy)

        optimizer_to(self.optimizer, torch.device('cpu'))
        with to_cpu(self.policy):
            with torch.no_grad():
                queue = fork_ctx.Queue()

                # Shared step counters для tqdm (один на эксперта)
                step_counters: Dict[str, fork_ctx.Value] = {
                    name: fork_ctx.Value('i', 0)
                    for name in self.experts
                }

                # Spawn ALL workers as background (daemon) processes
                total_workers = 0
                for expert_name, ctx in self.experts.items():
                    thread_batch = int(math.floor(ctx.min_batch_size / ctx.num_threads))
                    for worker_num in range(ctx.num_threads):
                        pid = total_workers + 1
                        profile_this = worker_num == 0 and self.epoch == 0
                        worker = fork_ctx.Process(
                            target=self.sample_worker,
                            args=(pid, queue, self.mp_done, thread_batch, expert_name),
                            kwargs={
                                "step_counter": step_counters[expert_name],
                                "enable_profiler": profile_this,
                            },
                            daemon=True,
                        )
                        worker.start()
                        total_workers += 1

                # Main process: только мониторинг progress bars
                pbars: Dict[str, tqdm] = {}
                for expert_name, ctx in self.experts.items():
                    pbars[expert_name] = tqdm(
                        total=ctx.min_batch_size,
                        desc=f"  Sampling [{expert_name}]",
                        unit="step",
                    )

                results: List = []
                while len(results) < total_workers:
                    for expert_name, counter in step_counters.items():
                        pbars[expert_name].n = min(counter.value, pbars[expert_name].total)
                        pbars[expert_name].refresh()

                    try:
                        result = queue.get(timeout=0.15)
                        results.append(result)
                    except Exception:
                        pass

                # Финальное обновление и закрытие bars
                for expert_name in step_counters:
                    pbars[expert_name].n = pbars[expert_name].total
                    pbars[expert_name].refresh()
                for pbar in pbars.values():
                    pbar.close()

                # Collect results per expert
                per_expert_memories: Dict[str, List[OBCMemory]] = {n: [] for n in self.experts}
                per_expert_loggers: Dict[str, List[OBCLogger]] = {n: [] for n in self.experts}

                for w_pid, w_expert, w_transfer, w_logger, w_profiler_data in results:
                    per_expert_memories[w_expert].append(OBCMemory.from_transfer_dict(w_transfer))
                    per_expert_loggers[w_expert].append(w_logger)
                    if w_profiler_data is not None:
                        self.profiler_reports[w_expert] = SamplingProfiler.from_dict(w_profiler_data)

        self.mp_done.set()
        optimizer_to(self.optimizer, self.device)

        # Merge per expert
        expert_batches: Dict[str, OBCBatch] = {}
        expert_loggers: Dict[str, OBCLogger] = {}

        for expert_name in self.experts:
            merged_memory = OBCMemory()
            for mem in per_expert_memories[expert_name]:
                merged_memory.extend(mem)

            batch = merged_memory.to_batch(gamma=self.gamma, tau=self.tau, device=None)
            expert_batches[expert_name] = batch
            expert_loggers[expert_name] = OBCLogger.merge(per_expert_loggers[expert_name])

        sample_time = time.time() - t_start
        for lg in expert_loggers.values():
            lg.sample_time = sample_time

        return expert_batches, expert_loggers

    # ------------------------------------------------------------------
    #  Update
    # ------------------------------------------------------------------

    def update_params(
        self,
        expert_batches: Dict[str, OBCBatch],
    ) -> Dict[str, Dict[str, float]]:
        """
        Обновляет параметры policy.

        Mini-batches от разных экспертов interleaved в рамках каждого PPO epoch.
        Каждый mini-batch использует loss-веса своего эксперта.

        Returns:
            Dict[expert_name → diagnostics_dict]
        """
        self.policy.train()

        # Подготовка per-expert данных на device
        expert_data = {}
        for expert_name, batch in expert_batches.items():
            ctx = self.experts[expert_name]
            expert_data[expert_name] = {
                "ctx": ctx,
                "states": batch.states.to(self.device),
                "actions": batch.student_actions.to(self.device),
                "returns": batch.returns.to(self.device),
                "advantages": batch.advantages.to(self.device),
                "fixed_log_probs": batch.log_probs.to(self.device),
                "old_values": batch.values.to(self.device),
                "expert_actions": batch.expert_actions.to(self.device) if ctx.use_expert else None,
                "obs_signatures": batch.obs_signatures,
                "action_signatures": batch.action_signatures,
                "batch_size": batch.states.shape[0],
            }

        # Accumulators per expert
        per_expert_losses: Dict[str, Dict[str, list]] = {
            n: {"ppo": [], "imitation": [], "value": [], "entropy": []}
            for n in self.experts
        }
        per_expert_diag: Dict[str, Dict[str, list]] = {
            n: {"ratios": [], "clip_fracs": [], "approx_kls": [], "grad_norms": []}
            for n in self.experts
        }

        # Общее число mini-batch updates для progress bar
        total_updates = 0
        for expert_name, ed in expert_data.items():
            n_batches = max(1, ed["batch_size"] // self.batch_size)
            total_updates += self.opt_num_epochs * n_batches

        pbar = tqdm(total=total_updates, desc="Update", unit="batch")

        for ppo_epoch in range(self.opt_num_epochs):
            # Строим interleaved расписание mini-batches
            schedule: List[Tuple[str, np.ndarray]] = []

            for expert_name, ed in expert_data.items():
                bs = ed["batch_size"]
                n_batches = max(1, bs // self.batch_size)
                indices = np.arange(bs)
                np.random.shuffle(indices)

                for i in range(n_batches):
                    start = i * self.batch_size
                    end = min((i + 1) * self.batch_size, bs)
                    schedule.append((expert_name, indices[start:end]))

            random.shuffle(schedule)

            for expert_name, batch_indices in schedule:
                ed = expert_data[expert_name]
                ctx = ed["ctx"]

                mini_states = ed["states"][batch_indices]
                mini_actions = ed["actions"][batch_indices]
                mini_returns = ed["returns"][batch_indices]
                mini_advantages = ed["advantages"][batch_indices]
                mini_fixed_lp = ed["fixed_log_probs"][batch_indices]
                mini_old_values = ed["old_values"][batch_indices]

                pred_actions, log_std, values = self.policy(
                    mini_states,
                    ed["obs_signatures"],
                    ed["action_signatures"],
                    expert_name=expert_name,
                )

                new_log_probs = self.policy._compute_log_prob(
                    mini_actions, pred_actions, log_std,
                )

                loss = torch.tensor(0.0, device=self.device)

                # --- PPO ---
                if ctx.ppo_weight > 0:
                    log_ratio = new_log_probs - mini_fixed_lp
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * mini_advantages
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon,
                    ) * mini_advantages
                    ppo_loss = -torch.min(surr1, surr2).mean()
                    loss = loss + ctx.ppo_weight * ppo_loss
                    per_expert_losses[expert_name]["ppo"].append(ppo_loss.item())

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean().item()
                        clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                        per_expert_diag[expert_name]["ratios"].append(ratio.mean().item())
                        per_expert_diag[expert_name]["clip_fracs"].append(clip_frac)
                        per_expert_diag[expert_name]["approx_kls"].append(approx_kl)

                # --- Imitation ---
                if ctx.use_expert and ctx.imitation_weight > 0:
                    mini_expert = ed["expert_actions"][batch_indices]
                    imitation_loss = nn.functional.mse_loss(pred_actions, mini_expert)
                    loss = loss + ctx.imitation_weight * imitation_loss
                    per_expert_losses[expert_name]["imitation"].append(imitation_loss.item())

                # --- Value ---
                if ctx.value_weight > 0:
                    vf_loss_unclipped = (values - mini_returns) ** 2
                    v_clipped = mini_old_values + torch.clamp(
                        values - mini_old_values,
                        -self.clip_epsilon,
                        self.clip_epsilon,
                    )
                    vf_loss_clipped = (v_clipped - mini_returns) ** 2
                    value_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()
                    loss = loss + ctx.value_weight * value_loss
                    per_expert_losses[expert_name]["value"].append(value_loss.item())

                # --- Entropy ---
                if ctx.entropy_weight > 0:
                    entropy = 0.5 * (1 + 1.8378770664093453) + log_std.mean()
                    entropy_loss = -entropy
                    loss = loss + ctx.entropy_weight * entropy_loss
                    per_expert_losses[expert_name]["entropy"].append(entropy_loss.item())

                # Apply expert loss scale
                loss = loss * ctx.loss_scale

                self.optimizer.zero_grad()
                loss.backward()

                if self.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.grad_clip,
                    )
                    per_expert_diag[expert_name]["grad_norms"].append(grad_norm.item())
                else:
                    total_norm = sum(
                        p.grad.norm().item() ** 2
                        for p in self.policy.parameters() if p.grad is not None
                    ) ** 0.5
                    per_expert_diag[expert_name]["grad_norms"].append(total_norm)

                self.optimizer.step()
                pbar.update(1)

        pbar.close()

        # Собираем диагностику per expert
        all_diagnostics: Dict[str, Dict[str, float]] = {}

        for expert_name in self.experts:
            ed = expert_data[expert_name]
            ctx = ed["ctx"]
            losses = per_expert_losses[expert_name]
            diag = per_expert_diag[expert_name]

            # Post-update diagnostics
            with torch.no_grad():
                bs = ed["batch_size"]
                diag_idx = np.arange(min(512, bs))
                diag_actions, diag_log_std, diag_values = self.policy(
                    ed["states"][diag_idx],
                    ed["obs_signatures"],
                    ed["action_signatures"],
                    expert_name=expert_name,
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

                adv = ed["advantages"]
                ret = ed["returns"]
                old_v = ed["old_values"]
                adv_mean = adv.mean().item()
                adv_std = adv.std().item()
                ret_mean = ret.mean().item()
                ret_std = ret.std().item()
                val_mean = old_v.mean().item()
                val_std = old_v.std().item()
                ret_var = ret.var().item()
                explained_var = 1 - (ret - old_v).var().item() / (ret_var + 1e-8)

            all_diagnostics[expert_name] = {
                "ppo_loss": float(np.mean(losses["ppo"])) if losses["ppo"] else 0.0,
                "imitation_loss": float(np.mean(losses["imitation"])) if losses["imitation"] else 0.0,
                "value_loss": float(np.mean(losses["value"])) if losses["value"] else 0.0,
                "entropy_loss": float(np.mean(losses["entropy"])) if losses["entropy"] else 0.0,
                "approx_kl": float(np.mean(diag["approx_kls"])) if diag["approx_kls"] else 0.0,
                "clip_frac": float(np.mean(diag["clip_fracs"])) if diag["clip_fracs"] else 0.0,
                "ratio_mean": float(np.mean(diag["ratios"])) if diag["ratios"] else 1.0,
                "grad_norm": float(np.mean(diag["grad_norms"])) if diag["grad_norms"] else 0.0,
                "explained_var": explained_var,
                "adv_mean": adv_mean,
                "adv_std": adv_std,
                "ret_mean": ret_mean,
                "ret_std": ret_std,
                "val_mean": val_mean,
                "val_std": val_std,
                "sigma_global": self.policy.log_sigma_global.item(),
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
            }

        # Update normalizer stats ONCE with the full batch data.
        # Must happen before the PPO loop — stats are frozen during updates
        # to avoid drift between new_log_probs and fixed_log_probs.
        with torch.no_grad():
            for expert_name, ed in expert_data.items():
                self.policy.obs_normalizer.update(
                    ed["obs_signatures"],
                    ed["states"],
                )


        return all_diagnostics

    # ------------------------------------------------------------------
    #  Main training loop
    # ------------------------------------------------------------------

    def optimize_policy(self) -> None:
        """Главный цикл обучения."""
        expert_names = list(self.experts.keys())
        logger.info(self._section_header(
            f"Training  |  experts: {', '.join(expert_names)}  |  epochs: {self.max_epochs}"
        ))

        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch

            # Pre-epoch: resample motions
            for expert_name, ctx in self.experts.items():
                if hasattr(ctx.wrapper.env, 'sample_motions') and epoch > 0:
                    if epoch % self.resampling_interval == 0:
                        ctx.wrapper.env.sample_motions()

            # Sample
            t_sample_start = time.time()
            expert_batches, expert_loggers = self.sample()
            t_sample = time.time() - t_sample_start

            # Update
            t_update_start = time.time()
            all_diagnostics = self.update_params(expert_batches)
            t_update = time.time() - t_update_start

            # Apply timing and losses to loggers
            total_steps_epoch = 0
            for expert_name, obc_lg in expert_loggers.items():
                obc_lg.sample_time = t_sample
                obc_lg.update_time = t_update
                diag = all_diagnostics[expert_name]
                obc_lg.set_update_losses(
                    ppo_loss=diag["ppo_loss"],
                    imitation_loss=diag["imitation_loss"],
                    value_loss=diag["value_loss"],
                    entropy_loss=diag["entropy_loss"],
                )
                total_steps_epoch += obc_lg.num_steps

            self.num_steps += total_steps_epoch

            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            if epoch % self.log_frequency == 0:
                self.log_train(epoch, expert_loggers, all_diagnostics)

            # Evaluation
            if self.eval_frequency > 0 and epoch > 0 and epoch % self.eval_frequency == 0:
                all_eval = self.evaluate()
                self.eval_checkpoint(epoch, all_eval)

            # Save current checkpoint
            if epoch > 0 and epoch % self.save_curr_frequency == 0:
                self.save_checkpoint(suffix="latest")

            # Save numbered checkpoint
            if epoch > 0 and epoch % self.save_frequency == 0:
                self.save_checkpoint(suffix=f"epoch_{epoch:05d}")

        # Final evaluation + save
        if self.eval_frequency > 0:
            logger.info("Final evaluation...")
            all_eval = self.evaluate()
            self.eval_checkpoint(epoch, all_eval)

        self.save_checkpoint(suffix="latest")
        logger.info("Training completed!")

        if self.use_wandb:
            self.wandb_logger.finish()

    def eval_checkpoint(
        self,
        epoch: int,
        all_eval: Dict[str, Dict[str, float]],
    ) -> None:
        """Сохраняет best-чекпоинты по результатам eval."""
        for expert_name, metrics in all_eval.items():
            if not metrics:
                continue

            ctx = self.experts[expert_name]

            im_loss = metrics.get("eval/imitation_loss", float('inf'))
            if ctx.use_expert and im_loss < self.best_eval_imitation_loss[expert_name]:
                self.best_eval_imitation_loss[expert_name] = im_loss
                self.save_checkpoint(suffix=f"best_im_loss_{expert_name}")

            ep_len = metrics.get("eval/mean_length", 0.0)
            if ep_len > self.best_eval_episode_avg_length[expert_name]:
                self.best_eval_episode_avg_length[expert_name] = ep_len
                self.save_checkpoint(suffix=f"best_ep_length_{expert_name}")

        if self.use_wandb:
            merged = {}
            for expert_name, metrics in all_eval.items():
                for k, v in metrics.items():
                    merged[f"{expert_name}/{k}"] = v
            self.wandb_logger.log_eval(epoch, merged)

    # ------------------------------------------------------------------
    #  Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _section_header(title: str, width: int = 60) -> str:
        """Формирует визуальный разделитель для логов."""
        return (
            "\n"
            "┌" + "─" * (width - 2) + "┐\n"
            "│  " + title.ljust(width - 4) + "│\n"
            "└" + "─" * (width - 2) + "┘"
        )

    def log_training_config(self) -> None:
        """Логирует конфигурацию обучения при старте."""
        logger.info(self._section_header("Training Configuration"))
        for expert_name, ctx in self.experts.items():
            logger.info(
                f"  {expert_name}: mode={ctx.training_mode.upper()}  "
                f"ppo={ctx.ppo_weight}  im={ctx.imitation_weight}  "
                f"val={ctx.value_weight}  ent={ctx.entropy_weight}  "
                f"scale={ctx.loss_scale}"
            )

    def log_train(
        self,
        epoch: int,
        expert_loggers: Dict[str, OBCLogger],
        all_diagnostics: Dict[str, Dict[str, float]],
    ) -> None:
        """Логирует метрики эпохи для каждого эксперта."""
        for expert_name, obc_logger in expert_loggers.items():
            ctx = self.experts[expert_name]
            logger.info(
                self._section_header(
                    f"Epoch {epoch}  |  {expert_name} ({ctx.training_mode.upper()})"
                )
            )

            log_str = obc_logger.get_log_str(epoch=epoch, expert_name=f"{self.exp_name}/{expert_name}")
            logger.info(log_str)

            d = all_diagnostics.get(expert_name, {})
            if d:
                logger.info(
                    f"  [{expert_name}] PPO diag: "
                    f"approx_kl={d.get('approx_kl', 0):.4f}  "
                    f"clip_frac={d.get('clip_frac', 0):.3f}  "
                    f"ratio={d.get('ratio_mean', 1):.4f}  "
                    f"grad_norm={d.get('grad_norm', 0):.3f}  "
                    f"σ_global={d.get('sigma_global', 0):.4f}  "
                    f"expl_var={d.get('explained_var', 0):.3f}  "
                    f"adv={d.get('adv_mean', 0):.4f}±{d.get('adv_std', 0):.4f}  "
                    f"ret={d.get('ret_mean', 0):.3f}±{d.get('ret_std', 0):.3f}  "
                    f"val={d.get('val_mean', 0):.3f}±{d.get('val_std', 0):.3f}"
                )
                logger.info(
                    f"  [{expert_name}] Policy out: "
                    f"act_mean={d.get('act_mean', 0):.4f}  "
                    f"act_std={d.get('act_std', 0):.4f}  "
                    f"|act|={d.get('act_abs_mean', 0):.4f}  "
                    f"act_range=[{d.get('act_min', 0):.3f}, {d.get('act_max', 0):.3f}]  "
                    f"logstd={d.get('logstd_mean', 0):.3f} [{d.get('logstd_min', 0):.3f}, {d.get('logstd_max', 0):.3f}]  "
                    f"val_post={d.get('val_post_mean', 0):.3f}±{d.get('val_post_std', 0):.3f}  "
                )

        # Per-expert sampling profiler — только эпоха 0
        if epoch == 0 and self.profiler_reports:
            for expert_name, profiler in self.profiler_reports.items():
                logger.info(self._section_header(
                    f"Sampling Profile  |  {expert_name}"
                ))
                logger.info(profiler.report())

        if self.use_wandb:
            self.wandb_logger.log_train(
                epoch=epoch,
                expert_loggers=expert_loggers,
                all_diagnostics=all_diagnostics,
                total_steps=self.num_steps,
            )

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """Оценивает policy на каждом эксперте отдельно."""
        logger.info(self._section_header("Evaluation"))
        all_eval: Dict[str, Dict[str, float]] = {}

        for expert_name, ctx in self.experts.items():
            valid_wrapper = self.valid_experts.get(expert_name)
            if valid_wrapper is None:
                all_eval[expert_name] = {}
                continue

            all_eval[expert_name] = self.evaluate_expert(expert_name, ctx, valid_wrapper)

        return all_eval

    def evaluate_expert(
        self,
        expert_name: str,
        ctx: ExpertContext,
        valid_wrapper,
    ) -> Dict[str, float]:
        """Evaluation для одного эксперта."""
        self.policy.eval()
        valid_parser = ObservationParser.from_env(valid_wrapper.env, self.history_len)

        episode_rewards = []
        episode_lengths = []
        episode_imitation_losses = []
        episode_value_losses = []

        for motion_id in tqdm(
            valid_wrapper.forward_motions(),
            total=valid_wrapper.num_motions,
            desc=f"Eval [{expert_name}]",
        ):
            obs, info = valid_wrapper.reset()
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
                        obs_ts, obs_sigs, act_sigs,
                        expert_name=expert_name,
                        deterministic=True,
                    )

                    if ctx.use_expert:
                        expert_action = valid_wrapper.get_expert_action(obs)
                        expert_action_t = torch.from_numpy(expert_action).float().to(self.device)
                        im_loss = ((action.squeeze(0) - expert_action_t) ** 2).mean().item()
                        step_imitation_losses.append(im_loss)

                    step_values.append(value.item())

                action_np = action.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = valid_wrapper.step(action_np)
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
                disc_returns = np.zeros(len(step_rewards), dtype=np.float64)
                running = 0.0
                for ri in reversed(range(len(step_rewards))):
                    running = step_rewards[ri] + self.gamma * running
                    disc_returns[ri] = running
                val_loss = np.mean((np.array(step_values) - disc_returns) ** 2)
                episode_value_losses.append(val_loss)

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
            f"Eval [{expert_name}]: "
            f"reward={metrics['eval/mean_reward']:.4f}±{metrics['eval/std_reward']:.4f}, "
            f"length={metrics['eval/mean_length']:.2f}±{metrics['eval/std_length']:.2f}"
            + (f", im_loss={metrics.get('eval/imitation_loss', 0):.4f}" if ctx.use_expert else "")
            + f", val_loss={metrics['eval/value_loss']:.4f}"
        )

        return metrics

    # ------------------------------------------------------------------
    #  Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, suffix: str = "latest") -> None:
        expert_modes = {n: ctx.training_mode for n, ctx in self.experts.items()}
        checkpoint = {
            "epoch": self.epoch,
            "num_steps": self.num_steps,
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "expert_names": list(self.experts.keys()),
            "expert_modes": expert_modes,
        }
        if self.scheduler is not None:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        path = os.path.join(self.output_dir, f"model_{suffix}.pth")
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, epoch: int) -> None:
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

        self.load_from_path(path, resume_training=True)

    def load_from_path(self, path: str, resume_training: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

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
            self.epoch = 0
            self.num_steps = 0

            # Normalizer stats keyed by semantic signature (e.g. "femur|r|position|x"),
            # so stats from source expert transfer naturally to target expert
            # for shared body parts. Only prune stats not needed by any new expert.
            ckpt_keys = set(self.policy.obs_normalizer.stats.keys())
            needed_keys = set()
            for ctx in self.experts.values():
                for sig in ctx.parser.obs_signatures:
                    needed_keys.add(self.policy.obs_normalizer._sig_key(sig))

            reused = ckpt_keys & needed_keys
            pruned = ckpt_keys - needed_keys
            fresh = needed_keys - ckpt_keys

            for key in pruned:
                del self.policy.obs_normalizer.stats[key]
            self.policy.obs_normalizer._invalidate_cache()

            logger.info(
                f"Transfer mode: normalizer stats — "
                f"{len(reused)} reused, {len(fresh)} new (no stats yet), "
                f"{len(pruned)} pruned (not in target experts)"
            )

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
            f"resuming from epoch {self.epoch}"
        )
