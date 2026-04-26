"""
Arnold Trainer — универсальный трейнер для Arnold.

Поддерживает три режима обучения:
1. OBC (On-Policy Behavior Cloning):
   ppo_weight=0, obc_imitation_weight>0
   → Чистая дистилляция из эксперта без PPO surrogate loss.

2. OBC-PPO:
   ppo_weight>0, obc_imitation_weight>0
   → Комбинация PPO и имитации — дистилляция с reinforcement learning.

3. PPO:
   ppo_weight>0, obc_imitation_weight=0
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
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from arnold.torch_model.transformer_policy import TransformerPolicy
from arnold.torch_model.sensorimotor_vocabulary import SensorimotorVocabulary
from arnold.observation_parser import ObservationParser, BodyGroup
from arnold.action_parser import ActionParser
from arnold.memory import OBCMemory, OBCBatch
from arnold.logger import OBCLogger
from arnold.wandb_logger import WandbLogger
from arnold.learning_utils import to_test, to_cpu, optimizer_to
from arnold.profiler import Profiler
from arnold.torch_model.dist_utils import safe_lrmvn_log_prob

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
    action_parser: ActionParser
    groups: List[BodyGroup]

    # Per-expert loss weights
    ppo_weight: float = 0.0
    obc_imitation_weight: float = 0.0
    entropy_weight: float = 0.0
    load_balance_weight: float = 0.0

    # Per-expert parallelism
    num_threads: int = 1
    min_batch_size: int = 10240
    vec_batch_size: Optional[int] = None

    # Per-expert loss scale (multiplier applied to all losses)
    loss_scale: float = 1.0

    # Action granulation (опционально)
    muscle_grouping: Optional[Any] = None

    # Offline BC weight (0 = disabled)
    bc_imitation_weight: float = 0.0

    @property
    def use_expert(self) -> bool:
        return self.obc_imitation_weight > 0

    @property
    def training_mode(self) -> str:
        if self.ppo_weight > 0 and self.obc_imitation_weight > 0:
            return "obc-ppo"
        elif self.obc_imitation_weight > 0:
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

    # Merge overrides: shared + per-mode (train_overrides / valid_overrides)
    mode_key = f"{mode}_overrides"
    expert_overrides = list(expert_entry.get("overrides", []))
    expert_overrides.extend(list(expert_entry.get(mode_key, [])))
    if overrides:
        expert_overrides.extend(overrides)

    if expert_type == "kinesis":
        from arnold.experts.kinesis_wrapper import KinesisWrapper
        model_type = expert_entry.get("model_type", "legs")
        return KinesisWrapper(
            cfg_path=expert_cfg_path,
            checkpoint_epoch=checkpoint_epoch,
            device="cpu",
            overrides=expert_overrides,
            mode=mode,
            model_type=model_type,
        )

    elif expert_type == "myohuman":
        from arnold.experts.myohuman_wrapper import MyoHumanWrapper
        simple = expert_entry.get("simple", False)
        return MyoHumanWrapper(
            cfg_path=expert_cfg_path,
            checkpoint_epoch=checkpoint_epoch,
            device="cpu",
            overrides=expert_overrides,
            mode=mode,
            simple=simple,
        )

    elif expert_type == "myokin":
        from arnold.experts.myokin_wrapper import MyoKinWrapper
        return MyoKinWrapper(
            cfg_path=expert_cfg_path,
            device="cpu",
            overrides=expert_overrides,
            mode=mode,
            teacher_checkpoint=expert_entry.get("teacher_checkpoint"),
            arm_action_override=expert_entry.get("arm_action_override", 0.0),
        )

    else:
        raise ValueError(
            f"Unknown expert type: '{expert_type}'. "
            f"Supported: 'kinesis', 'myohuman', 'myokin'"
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

        # DataParallel: device_ids=[0, 1] → обучение на 2 GPU
        device_ids = cfg.get("device_ids", None)
        base_device = str(device if device else cfg.device)
        if device_ids is not None and len(device_ids) >= 2 and "cuda" in base_device:
            self.device_ids = list(device_ids)
            self.device = torch.device(f"cuda:{self.device_ids[0]}")
        else:
            self.device_ids = None
            self.device = torch.device(device if device else cfg.device)

        # Global (parser, sampling)
        self.history_len = cfg.learning.history_len
        self.tokenizer_granularity = cfg.learning.tokenizer_granularity

        # Training (global defaults, per-expert overrides possible)
        self.batch_size = cfg.learning.batch_size
        self.learning_rate = cfg.learning.learning_rate
        self.value_learning_rate = cfg.learning.value_learning_rate
        self.weight_decay = cfg.learning.weight_decay
        self.gamma = cfg.learning.gamma
        self.tau = cfg.learning.tau
        self.clip_epsilon = cfg.learning.clip_epsilon
        self.value_clip = cfg.learning.value_clip
        self.opt_num_epochs = cfg.learning.opt_num_epochs
        self.grad_accum_steps = cfg.learning.grad_accum_steps
        self.grad_clip = cfg.learning.grad_clip
        self.value_grad_clip = cfg.learning.value_grad_clip
        self.max_epochs = cfg.learning.max_epochs
        self.use_scheduler = cfg.learning.use_scheduler
        self.use_compile = cfg.learning.use_compile
        self.use_bfloat16 = cfg.learning.use_bfloat16
        self.policy = None  # устанавливается в setup_policy()

        # Run (для доступа к submodules при DataParallel)
        self.policy_module = None  # устанавливается в setup_policy()

        # Run
        self.save_frequency = cfg.run.save_frequency
        self.save_curr_frequency = cfg.run.save_curr_frequency
        self.log_frequency = cfg.run.log_frequency
        self.output_dir = cfg.run.output_dir
        self.eval_frequency = cfg.run.eval_frequency
        self.train_eval_frequency = cfg.run.train_eval_frequency
        self.resampling_interval = cfg.run.resampling_interval
        self.finish_episodes = cfg.run.finish_episodes
        self.vectorized_eval = cfg.run.get("vectorized_eval", True)
        if not cfg.run.headless:
            self.vectorized_eval = False

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
        self.best_eval_mpjpe: Dict[str, float] = {}

        # Multiprocessing Event
        self.mp_done = fork_ctx.Event()

        # Persistent workers state (lazy init in _ensure_persistent_workers)
        self._persistent_state: Optional[dict] = None

        # Per-expert profiler reports (заполняется на epoch 0)
        self.profiler_reports: Dict[str, Profiler] = {}

        # Memory profiler report — собирается в update_params, печатается в log_train
        self._mem_profile_report: Optional[str] = None

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
            action_parser = ActionParser.from_env(wrapper.env)
            groups = parser.get_body_groups(self.tokenizer_granularity)

            learning_cfg = entry.get("learning", {})
            ppo_w = learning_cfg.get("ppo_weight") or 0.0
            im_w = learning_cfg.get("obc_imitation_weight") or 0.0
            bc_w = learning_cfg.get("bc_imitation_weight") or 0.0
            ent_w = learning_cfg.get("entropy_weight") or 0.0
            lb_w = learning_cfg.get("load_balance_weight") or 0.0
            loss_scale = learning_cfg.get("loss_scale") or 1.0
            mbs = learning_cfg.get("min_batch_size")
            n_threads = entry.get("num_threads") or 1
            vbs = entry.get("vec_batch_size", None)

            # Action granulation
            muscle_grouping = None
            action_gran = self._get_action_granulation()
            if action_gran != "none":
                muscle_grouping = action_parser.get_muscle_grouping(strategy=action_gran)
                logger.info(
                    f"  action granulation [{action_gran}]: "
                    f"{muscle_grouping.n_muscles} muscles -> {muscle_grouping.n_groups} groups"
                )
                for gid in muscle_grouping.group_order:
                    n_in_group = len(muscle_grouping.groups[gid])
                    logger.info(f"    {gid}: {n_in_group} muscles")

            ctx = ExpertContext(
                name=name,
                wrapper=wrapper,
                parser=parser,
                action_parser=action_parser,
                groups=groups,
                ppo_weight=ppo_w,
                obc_imitation_weight=im_w,
                bc_imitation_weight=bc_w,
                entropy_weight=ent_w,
                load_balance_weight=lb_w,
                num_threads=n_threads,
                min_batch_size=mbs,
                vec_batch_size=vbs,
                loss_scale=loss_scale,
                muscle_grouping=muscle_grouping,
            )
            self.experts[name] = ctx

            # Offline BC datasets
            bc_cfg = entry.get("bc", {})
            if bc_w > 0:
                if self.cfg.learning.policy != "lattice":
                    raise ValueError(
                        f"Expert '{name}': bc_imitation_weight > 0 requires policy=lattice, "
                        f"got policy={self.cfg.learning.policy}"
                    )
                bc_dir = bc_cfg.get("dataset_dir")
                if not bc_dir:
                    raise ValueError(
                        f"Expert '{name}': bc_imitation_weight > 0 but bc.dataset_dir is null"
                    )
                wrapper.load_bc_datasets(
                    dataset_dir=bc_dir,
                    batch_size=bc_cfg.get("batch_size", 256),
                    success_only=bc_cfg.get("success_only", False),
                    max_mpjpe_threshold=bc_cfg.get("max_mpjpe_threshold"),
                    mean_mpjpe_threshold=bc_cfg.get("mean_mpjpe_threshold"),
                )
                logger.info(f"  BC: weight={bc_w}")

            if (ctx.use_expert
                    and hasattr(wrapper, 'has_expert') and not wrapper.has_expert):
                logger.warning(
                    f"Expert '{name}': obc_imitation_weight > 0 but expert policy not loaded!"
                )

            logger.info(
                f"  mode={ctx.training_mode}  "
                f"obs_dim={wrapper.obs_dim}  act_dim={wrapper.action_dim}"
            )
            logger.info(
                f"  threads={n_threads}  min_batch={mbs}  "
                f"ppo={ppo_w}  im={im_w}  ent={ent_w}  lb={lb_w}  scale={loss_scale}"
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
            self.best_eval_mpjpe[name] = float('inf')

    def setup_policy(self) -> None:
        """Создаёт policy: TransformerPolicy или LatticePolicy."""
        logger.info("Setting up Arnold policy...")

        if self.cfg.learning.policy == "transformer":
            self._setup_transformer_policy()
        elif self.cfg.learning.policy == "lattice":
            self._setup_lattice_policy()
        elif self.cfg.learning.policy == "moe":
            self._setup_moe_policy()
        else:
            raise ValueError(
                f"Unknown policy: {self.cfg.learning.policy}. "
                f"Supported: transformer, lattice, moe"
            )

        self.policy.to(self.device)

        if self.use_compile:
            self.policy = torch.compile(self.policy, mode="reduce-overhead")
            logger.info("torch.compile applied (mode=reduce-overhead)")

        if self.device_ids is not None:
            self.policy = nn.DataParallel(self.policy, device_ids=self.device_ids)
            self.policy_module = self.policy.module
            logger.info(f"DataParallel enabled on GPUs {self.device_ids}")
        else:
            self.policy_module = self.policy

        logger.info(
            f"Arnold policy created ({self.policy}). "
            f"Parameters: {sum(p.numel() for p in self.policy.parameters()):,}"
        )

    def _get_action_granulation(self) -> str:
        """Получает стратегию action granulation из конфига."""
        if self.cfg.learning.policy != "transformer":
            return "none"
        transformer_cfg = self.cfg.learning.get("transformer", {})
        return transformer_cfg.get("action_granulation", "none")

    def _setup_transformer_policy(self) -> None:
        """TransformerPolicy с merged группами всех экспертов."""
        transformer_cfg = OmegaConf.to_container(
            self.cfg.learning.transformer, resolve=True
        )

        # Pop параметры, которые передаются как explicit kwargs, не через **transformer_cfg
        action_granulation = transformer_cfg.pop("action_granulation", "none")
        action_decoder_layers = transformer_cfg.pop("action_decoder_layers", None)

        self.vocab = SensorimotorVocabulary(embed_dim=transformer_cfg["embed_dim"])
        groups = {name: ctx.groups for name, ctx in self.experts.items()}
        max_action_dim = max(ctx.wrapper.action_dim for ctx in self.experts.values())

        # Action granulation: собираем groupings и signatures per expert
        action_groupings = None
        action_sigs_by_expert = None
        if action_granulation != "none":
            action_groupings = {}
            action_sigs_by_expert = {}
            for name, ctx in self.experts.items():
                if ctx.muscle_grouping is None:
                    raise ValueError(
                        f"action_granulation={action_granulation!r} but expert "
                        f"'{name}' has no muscle_grouping. "
                        f"Check that setup_experts() ran before setup_policy()."
                    )
                action_groupings[name] = ctx.muscle_grouping
                action_sigs_by_expert[name] = ctx.action_parser.action_signatures

        self.policy = TransformerPolicy(
            vocab=self.vocab,
            groups=groups,
            history_len=self.history_len,
            max_action_dim=max_action_dim,
            action_granulation=action_granulation,
            action_groupings=action_groupings,
            action_signatures_by_expert=action_sigs_by_expert,
            action_decoder_layers=action_decoder_layers,
            **transformer_cfg,
        )

    def _setup_lattice_policy(self) -> None:
        """LatticePolicy — MLP с полной ковариацией (как в Kinesis). Только single-expert."""
        from arnold.torch_model.policy_lattice import LatticePolicy

        if len(self.experts) > 1:
            raise ValueError(
                f"Lattice policy поддерживает только одного эксперта, "
                f"сейчас: {list(self.experts.keys())}. "
                "Используйте policy=transformer для multi-expert."
            )

        ctx = next(iter(self.experts.values()))
        # Kinesis: только текущий state, без истории (history_len=1)
        state_dim = ctx.parser.n_obs_elements
        action_dim = ctx.wrapper.action_dim

        lattice_cfg = OmegaConf.to_container(
            self.cfg.learning.lattice, resolve=True
        )

        self.policy = LatticePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            **lattice_cfg
        )

    def _setup_moe_policy(self) -> None:
        """MoE-Lattice Policy — Shared Expert + N Routed Experts, Low-Rank Covariance."""
        from arnold.torch_model.policy_moe import MoELatticePolicy

        if len(self.experts) > 1:
            raise ValueError(
                f"MoE policy поддерживает только одного эксперта, "
                f"сейчас: {list(self.experts.keys())}. "
                "Используйте policy=transformer для multi-expert."
            )

        ctx = next(iter(self.experts.values()))
        state_dim = ctx.parser.n_obs_elements
        action_dim = ctx.wrapper.action_dim

        moe_cfg = OmegaConf.to_container(self.cfg.learning.moe, resolve=True)

        latent_dim = moe_cfg["shared_expert_units"][-1]
        num_experts = moe_cfg["num_experts"]
        top_k = moe_cfg["top_k"]
        alpha = moe_cfg["alpha"]

        routing_mode = "soft (all experts)" if top_k == num_experts else f"sparse top_k={top_k}"
        logger.info(
            f"MoE setup:\n"
            f"  state_dim={state_dim}  action_dim={action_dim}  latent_dim={latent_dim}\n"
            f"  alpha={alpha} (shared) / {1 - alpha:.2f} (routed)\n"
            f"  routing: {routing_mode} / {num_experts} experts  "
            f"(~{top_k / num_experts:.0%} of pool per sample)\n"
            f"  cov_factor: α·W_shared + (1-α)·Σ(w_i·W_i) * latent_std  "
            f"[{action_dim}×{latent_dim}]"
        )

        self.policy = MoELatticePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            **moe_cfg,
        )

    def setup_optimizer(self) -> None:
        self.policy_optimizer = torch.optim.AdamW(
            self.policy_module.policy_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.value_optimizer = torch.optim.AdamW(
            self.policy_module.value_parameters(),
            lr=self.value_learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.use_scheduler:
            self.policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.policy_optimizer,
                T_max=self.max_epochs,
                eta_min=self.learning_rate * 0.1,
            )
            self.value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.value_optimizer,
                T_max=self.max_epochs,
                eta_min=self.value_learning_rate * 0.1,
            )
        else:
            self.policy_scheduler = None
            self.value_scheduler = None

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
        enable_profiler=True — worker включает time profiling и отправляет
        данные через queue (только один worker на эксперта, epoch 0).
        """
        self.seed_worker(pid)

        ctx = self.experts[expert_name]
        wrapper = ctx.wrapper
        worker_parser = ObservationParser.from_env(wrapper.env, self.history_len)
        worker_action_parser = ActionParser.from_env(wrapper.env)

        memory = OBCMemory()
        obc_logger = OBCLogger()

        if enable_profiler:
            self.policy_module.profiler.time_enabled = True

        if not ctx.use_expert:
            zero_expert = torch.zeros(wrapper.action_dim, dtype=torch.float32)

        try:
            while obc_logger.num_steps < min_batch_size:
                obs, info = wrapper.reset()
                worker_parser.reset(obs)

                p = self.policy_module.profiler
                for t in range(10000):
                    with p.section("parser.get_obs"):
                        obs_ts, obs_sigs = worker_parser.get_observation(torch.device("cpu"))
                        act_sigs = worker_action_parser.action_signatures

                    with p.section("policy.forward"):
                        with torch.no_grad():
                            student_action, log_prob, value = self.policy_module.get_action(
                                obs_ts, obs_sigs, act_sigs,
                                expert_name=expert_name,
                                deterministic=False,
                            )

                    if ctx.use_expert:
                        with p.section("expert.get_action"):
                            expert_action = wrapper.get_expert_action(obs)
                            expert_action_t = torch.from_numpy(expert_action).float()
                    else:
                        expert_action_t = zero_expert

                    student_action_np = student_action.squeeze(0).cpu().numpy()
                    with p.section("env.step"):
                        next_obs, reward, terminated, truncated, info = wrapper.step(student_action_np)
                    done = terminated or truncated

                    with p.section("memory.append"):
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

                    if step_counter is not None:
                        with step_counter.get_lock():
                            step_counter.value += 1

                    if done:
                        break

                    with p.section("parser.update"):
                        worker_parser.update(next_obs)
                    obs = next_obs

                obc_logger.end_episode()

        except Exception as e:
            import traceback
            print(f"Worker {pid} [{expert_name}] failed: {e}")
            traceback.print_exc()

        finally:
            p = self.policy_module.profiler
            profiler_data = p.to_dict() if p.time_enabled else None
            p.time_enabled = False
            queue.put([pid, expert_name, memory.to_transfer_dict(), obc_logger, profiler_data])
            mp_done.wait()

    def sample(self) -> Tuple[Dict[str, OBCBatch], Dict[str, OBCLogger]]:
        """
        Собирает траектории от ВСЕХ экспертов полностью параллельно.

        ВСЕ workers запускаются как background processes (num_threads на каждого
        эксперта). Main process только мониторит shared counters и обновляет
        N tqdm progress bars одновременно — без потери производительности.

        На epoch 0 первый worker каждого эксперта включает time profiling
        и передаёт результаты обратно — per-expert profiling.

        Returns:
            expert_batches: Dict[expert_name -> OBCBatch]
            expert_loggers: Dict[expert_name -> OBCLogger]
        """
        # Vectorized: env workers on CPU, batched inference on GPU/MPS
        if self.device.type in ('cuda', 'mps'):
            return self._sample_vectorized_persistent()

        t_start = time.time()
        self.mp_done.clear()
        to_test(self.policy)

        optimizer_to(self.policy_optimizer, torch.device('cpu'))
        optimizer_to(self.value_optimizer, torch.device('cpu'))
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
                workers: List[fork_ctx.Process] = []
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
                        workers.append(worker)
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

                    # Check for dead workers (segfault, SIGKILL, etc.)
                    dead_workers = [
                        w for w in workers
                        if not w.is_alive() and w.exitcode is not None and w.exitcode != 0
                    ]
                    finished_not_reported = sum(
                        1 for w in workers if not w.is_alive()
                    ) - len(results)
                    if finished_not_reported > 0 and len(dead_workers) > 0:
                        exit_codes = {w.pid: w.exitcode for w in dead_workers}
                        raise RuntimeError(
                            f"Sampling workers crashed (exit codes: {exit_codes}). "
                            f"This usually means a segfault in forked process. "
                            f"Check worker logs above for details."
                        )

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
                        self.profiler_reports[w_expert] = Profiler.from_dict(w_profiler_data)

        self.mp_done.set()
        optimizer_to(self.policy_optimizer, self.device)
        optimizer_to(self.value_optimizer, self.device)

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
    #  Vectorized Sampling
    # ------------------------------------------------------------------

    def _vec_env_worker_persistent(
        self,
        expert_name: str,
        worker_id: int,
        obs_buf: torch.Tensor,
        action_buf: torch.Tensor,
        expert_action_buf: torch.Tensor,
        reward_val,
        done_val,
        obs_ready,
        action_ready,
        stop_flag,
        shutdown_flag,
        epoch_start,
        epoch_val,
        resample_flag,
        result_queue,
        step_counter,
        send_sigs: bool,
    ) -> None:
        """
        Persistent env worker for vectorized sampling.

        Outer loop: waits for epoch_start, runs inner sampling loop, sends
        logger/stats, loops. Exits when shutdown_flag is set.
        """
        ctx = self.experts[expert_name]
        wrapper = ctx.wrapper
        parser = ObservationParser.from_env(wrapper.env, self.history_len)
        action_parser = ActionParser.from_env(wrapper.env)

        sigs_sent = False

        try:
            while True:
                # Wait for next epoch or shutdown
                while not epoch_start.wait(timeout=1.0):
                    if shutdown_flag.value:
                        return
                epoch_start.clear()

                if shutdown_flag.value:
                    return

                # Re-seed per epoch for reproducibility
                current_epoch = epoch_val.value
                seed = random.randint(0, 5000) * worker_id + current_epoch
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                # Resample motions if main process requested it
                if resample_flag.value and hasattr(wrapper.env, 'sample_motions'):
                    wrapper.env.sample_motions()

                obc_logger = OBCLogger()

                # Reset env at epoch start
                obs, info = wrapper.reset()
                parser.reset(obs)
                obs_ts, obs_sigs = parser.get_observation(torch.device("cpu"))
                act_sigs = action_parser.action_signatures

                if send_sigs and not sigs_sent:
                    result_queue.put(("sigs", expert_name, obs_sigs, act_sigs))
                    sigs_sent = True

                obs_buf.copy_(obs_ts.squeeze(0))
                if ctx.use_expert:
                    ea = wrapper.get_expert_action(obs)
                    expert_action_buf.copy_(torch.from_numpy(ea).float())

                done_val.value = 0
                reward_val.value = 0.0
                step_counter.value = 0
                obs_ready.set()

                # Inner sampling loop (identical to ephemeral)
                try:
                    while True:
                        action_ready.wait()
                        action_ready.clear()

                        if stop_flag.value:
                            break

                        action_np = action_buf.numpy().copy()
                        next_obs, reward, terminated, truncated, info = wrapper.step(action_np)
                        done = terminated or truncated

                        reward_val.value = float(reward)
                        done_val.value = 1 if done else 0

                        obc_logger.step(
                            reward=reward,
                            info=info if isinstance(info, dict) else None,
                        )
                        with step_counter.get_lock():
                            step_counter.value += 1

                        if done:
                            obc_logger.end_episode()
                            obs, info = wrapper.reset()
                            parser.reset(obs)
                        else:
                            parser.update(next_obs)
                            obs = next_obs

                        obs_ts, _ = parser.get_observation(torch.device("cpu"))
                        obs_buf.copy_(obs_ts.squeeze(0))
                        if ctx.use_expert:
                            ea = wrapper.get_expert_action(obs)
                            expert_action_buf.copy_(torch.from_numpy(ea).float())

                        obs_ready.set()

                except Exception as e:
                    import traceback
                    print(f"PersistentWorker [{expert_name}#{worker_id}] inner loop failed: {e}")
                    traceback.print_exc()

                # Send logger + termination stats for this epoch
                term_stats = None
                if hasattr(wrapper.env, "get_termination_stats"):
                    term_stats = wrapper.env.get_termination_stats()
                result_queue.put(("logger", expert_name, obc_logger, term_stats))

        except Exception as e:
            import traceback
            print(f"PersistentWorker [{expert_name}#{worker_id}] outer loop failed: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    #  Persistent Worker Management
    # ------------------------------------------------------------------

    def _ensure_persistent_workers(self) -> None:
        """Lazily spawn persistent workers on first call. No-op if already alive."""
        if self._persistent_state is not None:
            return

        result_queue = fork_ctx.Queue()
        stop_flag = fork_ctx.Value('i', 0)
        shutdown_flag = fork_ctx.Value('i', 0)
        epoch_val = fork_ctx.Value('i', 0)
        resample_flag = fork_ctx.Value('i', 0)

        class _WH:
            """Persistent worker handle."""
            __slots__ = (
                'expert_name', 'obs_buf', 'action_buf', 'expert_action_buf',
                'reward_val', 'done_val', 'obs_ready', 'action_ready',
                'epoch_start', 'proc', 'active',
            )

        workers_by_expert: Dict[str, List] = {n: [] for n in self.experts}
        all_workers: List = []

        for expert_name, ctx in self.experts.items():
            obs_shape = (ctx.parser.n_obs_elements, self.history_len)
            action_dim = ctx.wrapper.action_dim
            n_workers = ctx.vec_batch_size or ctx.num_threads

            for w_idx in range(n_workers):
                wh = _WH()
                wh.expert_name = expert_name
                wh.obs_buf = torch.zeros(obs_shape, dtype=torch.float32).share_memory_()
                wh.action_buf = torch.zeros(action_dim, dtype=torch.float32).share_memory_()
                wh.expert_action_buf = torch.zeros(action_dim, dtype=torch.float32).share_memory_()
                wh.reward_val = fork_ctx.Value('d', 0.0)
                wh.done_val = fork_ctx.Value('i', 0)
                wh.obs_ready = fork_ctx.Event()
                wh.action_ready = fork_ctx.Event()
                wh.epoch_start = fork_ctx.Event()
                wh.active = True

                global_wid = len(all_workers) + 1

                wh.proc = fork_ctx.Process(
                    target=self._vec_env_worker_persistent,
                    args=(
                        expert_name,
                        global_wid,
                        wh.obs_buf, wh.action_buf, wh.expert_action_buf,
                        wh.reward_val, wh.done_val,
                        wh.obs_ready, wh.action_ready,
                        stop_flag, shutdown_flag,
                        wh.epoch_start, epoch_val,
                        resample_flag,
                        result_queue,
                        fork_ctx.Value('i', 0),  # step_counter per worker
                        w_idx == 0,  # send_sigs
                    ),
                    daemon=True,
                )
                workers_by_expert[expert_name].append(wh)
                all_workers.append(wh)

        for wh in all_workers:
            wh.proc.start()

        total_workers = len(all_workers)
        workers_per_expert = {n: len(ws) for n, ws in workers_by_expert.items()}
        logger.info(
            f"Persistent vectorized workers spawned: {total_workers} "
            f"({workers_per_expert})"
        )

        # Collect signatures from first epoch trigger
        # Signal all workers to start their first epoch
        epoch_val.value = self.epoch
        for wh in all_workers:
            wh.epoch_start.set()

        sigs_by_expert: Dict[str, Tuple] = {}
        while len(sigs_by_expert) < len(self.experts):
            msg = result_queue.get(timeout=30)
            if msg[0] == "sigs":
                sigs_by_expert[msg[1]] = (msg[2], msg[3])

        self._persistent_state = {
            "result_queue": result_queue,
            "stop_flag": stop_flag,
            "shutdown_flag": shutdown_flag,
            "epoch_val": epoch_val,
            "resample_flag": resample_flag,
            "workers_by_expert": workers_by_expert,
            "all_workers": all_workers,
            "sigs_by_expert": sigs_by_expert,
            "first_epoch": True,  # first epoch already triggered
        }

    def _shutdown_persistent_workers(self) -> None:
        """Shutdown all persistent workers and clean up."""
        if self._persistent_state is None:
            return

        state = self._persistent_state
        state["shutdown_flag"].value = 1

        # Wake workers so they see shutdown_flag
        for wh in state["all_workers"]:
            wh.epoch_start.set()
            wh.action_ready.set()

        for wh in state["all_workers"]:
            wh.proc.join(timeout=10)
            if wh.proc.is_alive():
                wh.proc.terminate()

        self._persistent_state = None
        logger.info("Persistent workers shut down.")

    def _sample_vectorized_persistent(self) -> Tuple[Dict[str, OBCBatch], Dict[str, OBCLogger]]:
        """
        Vectorized sampling with persistent workers.

        Workers are already alive (spawned by _ensure_persistent_workers).
        This method triggers a new epoch, runs the lockstep inference loop,
        and collects results — without process spawn overhead.
        """
        self._ensure_persistent_workers()
        state = self._persistent_state

        t_start = time.time()
        self.policy.eval()

        optimizer_to(self.policy_optimizer, torch.device('cpu'))
        optimizer_to(self.value_optimizer, torch.device('cpu'))

        prof = Profiler(
            device=str(self.device) if self.device.type == 'cuda' else None,
        )
        prof.time_enabled = (self.epoch == 0)

        old_profiler = getattr(self.policy_module, 'profiler', None)
        if hasattr(self.policy_module, 'set_profiler'):
            self.policy_module.set_profiler(prof)

        result_queue = state["result_queue"]
        stop_flag = state["stop_flag"]
        workers_by_expert = state["workers_by_expert"]
        all_workers = state["all_workers"]
        sigs_by_expert = state["sigs_by_expert"]

        total_workers = len(all_workers)

        # Reset stop flag and activate all workers
        stop_flag.value = 0
        for wh in all_workers:
            wh.active = True

        # Signal epoch start (skip on first epoch — already triggered in _ensure)
        if not state["first_epoch"]:
            state["epoch_val"].value = self.epoch
            for wh in all_workers:
                wh.epoch_start.set()
        state["first_epoch"] = False

        workers_per_expert = {n: len(ws) for n, ws in workers_by_expert.items()}
        logger.info(
            f"Persistent sampling: {total_workers} workers "
            f"({workers_per_expert}), inference on {self.device}, "
            f"finish_episodes={self.finish_episodes}"
        )

        # Wait for initial obs from all workers
        def _wait_obs_active():
            for wh in all_workers:
                if not wh.active:
                    continue
                while not wh.obs_ready.wait(timeout=2.0):
                    if not wh.proc.is_alive():
                        raise RuntimeError(
                            f"PersistentWorker [{wh.expert_name}] died "
                            f"(exit code {wh.proc.exitcode})"
                        )
                wh.obs_ready.clear()

        _wait_obs_active()

        # Per-worker memory tracking
        worker_mems: Dict[int, OBCMemory] = {id(w): OBCMemory() for w in all_workers}
        worker_prev: Dict[int, Optional[dict]] = {id(w): None for w in all_workers}

        per_expert_steps: Dict[str, int] = {n: 0 for n in self.experts}
        target_steps = {n: ctx.min_batch_size for n, ctx in self.experts.items()}

        pbars = {
            n: tqdm(total=t, desc=f"  Sampling [{n}]", unit="step")
            for n, t in target_steps.items()
        }

        device = self.device
        device_type = "cuda" if device.type == "cuda" else "cpu"
        use_amp = self.use_bfloat16 and device.type == "cuda"
        has_prev = False
        draining = False

        # Main inference loop (identical to ephemeral)
        with torch.no_grad():
            while True:
                if has_prev:
                    with prof.section("store_memory"):
                        for expert_name, workers in workers_by_expert.items():
                            obs_sigs, act_sigs = sigs_by_expert[expert_name]

                            for wh in workers:
                                if not wh.active:
                                    continue
                                wid = id(wh)
                                prev = worker_prev[wid]
                                if prev is None:
                                    continue
                                mem = worker_mems[wid]

                                mem.states.append(prev["obs"])
                                mem.obs_signatures.append(obs_sigs)
                                mem.action_signatures.append(act_sigs)
                                mem.student_actions.append(prev["action"])
                                mem.expert_actions.append(prev["expert_action"])
                                mem.rewards.append(wh.reward_val.value)
                                mem.values.append(prev["value"])
                                mem.masks.append(0.0 if wh.done_val.value else 1.0)
                                mem.log_probs.append(prev["log_prob"])
                                per_expert_steps[expert_name] += 1

                    for n, pbar in pbars.items():
                        pbar.n = min(per_expert_steps[n], pbar.total)
                        pbar.refresh()

                    if draining:
                        for wh in all_workers:
                            if wh.active and wh.done_val.value:
                                wh.active = False
                        if not any(wh.active for wh in all_workers):
                            break

                    target_reached = all(
                        per_expert_steps[n] >= target_steps[n]
                        for n in self.experts
                    )
                    if target_reached:
                        if not self.finish_episodes:
                            break
                        if not draining:
                            draining = True
                            for wh in all_workers:
                                if wh.done_val.value:
                                    wh.active = False
                            if not any(wh.active for wh in all_workers):
                                break

                with prof.section("inference"):
                    for expert_name, workers in workers_by_expert.items():
                        obs_sigs, act_sigs = sigs_by_expert[expert_name]
                        ctx = self.experts[expert_name]

                        active_workers = [wh for wh in workers if wh.active]
                        if not active_workers:
                            continue

                        with prof.section("stack_obs"):
                            batch_obs = torch.stack(
                                [wh.obs_buf.clone() for wh in active_workers]
                            )

                        with prof.section("gpu_forward"):
                            with prof.section("to_device"):
                                batch_obs_dev = batch_obs.to(device)
                            with prof.section("policy_forward"):
                                with torch.autocast(
                                    device_type=device_type,
                                    dtype=torch.bfloat16,
                                    enabled=use_amp,
                                ):
                                    actions, log_probs, values = \
                                        self.policy_module.get_action(
                                            batch_obs_dev,
                                            obs_sigs, act_sigs,
                                            expert_name=expert_name,
                                            deterministic=False,
                                        )
                            with prof.section("to_cpu"):
                                actions_cpu = actions.cpu()
                                log_probs_cpu = log_probs.cpu()
                                values_cpu = values.cpu()

                        with prof.section("scatter"):
                            zero_ea = torch.zeros(ctx.wrapper.action_dim)
                            for i, wh in enumerate(active_workers):
                                if ctx.use_expert:
                                    ea = wh.expert_action_buf.clone()
                                else:
                                    ea = zero_ea
                                worker_prev[id(wh)] = {
                                    "obs": wh.obs_buf.clone(),
                                    "action": actions_cpu[i],
                                    "log_prob": log_probs_cpu[i],
                                    "value": values_cpu[i],
                                    "expert_action": ea,
                                }
                                wh.action_buf.copy_(actions_cpu[i])

                has_prev = True

                with prof.section("signal_workers"):
                    for wh in all_workers:
                        if wh.active:
                            wh.action_ready.set()

                with prof.section("wait_env_step"):
                    _wait_obs_active()

        # Stop workers (they'll send logger via queue, then wait for next epoch)
        for pbar in pbars.values():
            pbar.n = pbar.total
            pbar.refresh()
            pbar.close()

        stop_flag.value = 1
        for wh in all_workers:
            wh.action_ready.set()

        # Collect loggers from workers
        per_expert_loggers: Dict[str, List[OBCLogger]] = {n: [] for n in self.experts}
        per_expert_term_stats: Dict[str, list] = {n: [] for n in self.experts}
        collected = 0
        while collected < total_workers:
            try:
                msg = result_queue.get(timeout=10)
                if msg[0] == "logger":
                    per_expert_loggers[msg[1]].append(msg[2])
                    if msg[3] is not None:
                        per_expert_term_stats[msg[1]].append(msg[3])
                    collected += 1
            except Exception:
                break

        # Restore profiler
        if old_profiler is not None and hasattr(self.policy_module, 'set_profiler'):
            self.policy_module.set_profiler(old_profiler)

        optimizer_to(self.policy_optimizer, self.device)
        optimizer_to(self.value_optimizer, self.device)

        # Merge per expert
        expert_batches: Dict[str, OBCBatch] = {}
        expert_loggers: Dict[str, OBCLogger] = {}

        for expert_name, workers in workers_by_expert.items():
            merged = OBCMemory()
            for wh in workers:
                mem = worker_mems[id(wh)]
                if len(mem) > 0 and mem.masks[-1] != 0.0:
                    mem.masks[-1] = 0.0
                merged.extend(mem)

            batch = merged.to_batch(gamma=self.gamma, tau=self.tau, device=None)
            expert_batches[expert_name] = batch
            expert_loggers[expert_name] = OBCLogger.merge(
                per_expert_loggers.get(expert_name, [])
            )

        # Merge termination stats
        for expert_name, stats_list in per_expert_term_stats.items():
            if not stats_list:
                continue
            ctx = self.experts[expert_name]
            env = ctx.wrapper.env
            if not hasattr(env, "_termination_counts"):
                env._termination_counts = {}
                env._termination_dists = {}
            for worker_stats in stats_list:
                for body, s in worker_stats.items():
                    if body not in env._termination_counts:
                        env._termination_counts[body] = 0
                        env._termination_dists[body] = []
                    env._termination_counts[body] += s["caused"]
                    if s["mean_dist"] > 0 and s["caused"] > 0:
                        env._termination_dists[body].extend([s["mean_dist"]] * s["caused"])

        if prof.time_enabled:
            self.profiler_reports["vectorized_sampling"] = prof

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
        debug_memory: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Обновляет параметры policy.

        Mini-batches от разных экспертов interleaved в рамках каждого PPO epoch.
        Каждый mini-batch использует loss-веса своего эксперта.

        debug_memory=True: профилирует потребление GPU-памяти первого mini-batch
                           и печатает деревянный отчёт через logger.info.

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
            n: {"ppo": [], "imitation": [], "imitation_per_sample": [], "value": [], "entropy": [], "sigma": [], "load_balance": [], "bc": []}
            for n in self.experts
        }
        per_expert_diag: Dict[str, Dict[str, list]] = {
            n: {"ratios": [], "clip_fracs": [], "value_clip_fracs": [], "approx_kls": [], "grad_norms": [], "value_grad_norms": []}
            for n in self.experts
        }

        # Общее число mini-batch forwards для progress bar
        total_updates = 0
        for expert_name, ed in expert_data.items():
            n_batches = max(1, ed["batch_size"] // self.batch_size)
            total_updates += self.opt_num_epochs * n_batches
        # Log effective optimizer steps

        pbar = tqdm(total=total_updates, desc="Update", unit="batch")

        # Memory profiling: включается на первый mini-batch, потом выключается
        _mem_prof_done: bool = False

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

            accum_step = 0
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()

            for sched_idx, (expert_name, batch_indices) in enumerate(schedule):
                ed = expert_data[expert_name]
                ctx = ed["ctx"]

                # Включаем memory profiling для первого mini-batch
                _profile_this = debug_memory and not _mem_prof_done
                prof = self.policy_module.profiler
                if _profile_this:
                    prof.device = self.device
                    prof.mem_enabled = True
                    prof.reset()

                mini_states = ed["states"][batch_indices]
                mini_actions = ed["actions"][batch_indices]
                mini_returns = ed["returns"][batch_indices]
                mini_advantages = ed["advantages"][batch_indices]
                mini_fixed_lp = ed["fixed_log_probs"][batch_indices]
                mini_old_values = ed["old_values"][batch_indices]

                use_autocast = self.use_bfloat16 and self.device.type == "cuda"

                with prof.section(f"mini_batch  expert={expert_name}"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                        with prof.section("policy.forward"):
                            if prof.mem_enabled and self.device_ids is not None:
                                # DataParallel реплики не видят profiler →
                                # при профилировании вызываем policy_module напрямую
                                policy_out = self.policy_module.forward(
                                    mini_states,
                                    ed["obs_signatures"],
                                    ed["action_signatures"],
                                    expert_name=expert_name,
                                )
                            else:
                                policy_out = self.policy(
                                    mini_states,
                                    ed["obs_signatures"],
                                    ed["action_signatures"],
                                    expert_name=expert_name,
                                )
                        pred_mean, cov_factor, diag_std, values, latent_std = policy_out[:5]
                        forward_lb_loss = policy_out[5] if len(policy_out) > 5 else None

                        with prof.section("build_dist"):
                            dist = self.policy_module.build_action_dist(
                                pred_mean, cov_factor, diag_std,
                            )

                        with prof.section("compute_log_prob"):
                            new_log_probs = safe_lrmvn_log_prob(dist, mini_actions.float()).unsqueeze(-1)

                        policy_loss = torch.tensor(0.0, device=self.device)
                        value_loss_total = torch.tensor(0.0, device=self.device)

                        # --- PPO ---
                        if ctx.ppo_weight > 0:
                            with prof.section("ppo_loss"):
                                log_ratio = new_log_probs - mini_fixed_lp
                                ratio = torch.exp(log_ratio)
                                surr1 = ratio * mini_advantages
                                surr2 = torch.clamp(
                                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon,
                                ) * mini_advantages
                                ppo_loss = -torch.min(surr1, surr2).mean()
                            policy_loss = policy_loss + ctx.ppo_weight * ppo_loss
                            per_expert_losses[expert_name]["ppo"].append(ppo_loss.item())

                            with torch.no_grad():
                                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                                clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                                per_expert_diag[expert_name]["ratios"].append(ratio.mean().item())
                                per_expert_diag[expert_name]["clip_fracs"].append(clip_frac)
                                per_expert_diag[expert_name]["approx_kls"].append(approx_kl)

                        # --- OBC Imitation ---
                        if ctx.use_expert and ctx.obc_imitation_weight > 0:
                            with prof.section("imitation_loss"):
                                mini_expert = ed["expert_actions"][batch_indices]
                                per_sample = ((pred_mean - mini_expert) ** 2).mean(dim=1)
                                imitation_loss = per_sample.mean()
                            policy_loss = policy_loss + ctx.obc_imitation_weight * imitation_loss
                            per_expert_losses[expert_name]["imitation"].append(imitation_loss.item())
                            per_expert_losses[expert_name]["imitation_per_sample"].append(imitation_loss.item())

                        # --- Value ---
                        with prof.section("value_loss"):
                            vf_loss_unclipped = (values - mini_returns) ** 2
                            if self.value_clip:
                                v_clipped = mini_old_values + torch.clamp(
                                    values - mini_old_values,
                                    -self.clip_epsilon,
                                    self.clip_epsilon,
                                )
                                vf_loss_clipped = (v_clipped - mini_returns) ** 2
                                value_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()
                            else:
                                value_loss = 0.5 * vf_loss_unclipped.mean()
                        value_loss_total = value_loss_total + value_loss
                        per_expert_losses[expert_name]["value"].append(value_loss.item())
                        if self.value_clip:
                            with torch.no_grad():
                                value_clip_frac = (
                                    (values - mini_old_values).abs() > self.clip_epsilon
                                ).float().mean().item()
                                per_expert_diag[expert_name]["value_clip_fracs"].append(value_clip_frac)

                        # --- Entropy ---
                        if ctx.entropy_weight > 0:
                            with prof.section("entropy"):
                                entropy = dist.entropy().mean()
                            entropy_loss = -entropy
                            policy_loss = policy_loss + ctx.entropy_weight * entropy_loss
                            per_expert_losses[expert_name]["entropy"].append(entropy_loss.item())

                        # --- MoE load balancing ---
                        if (ctx.load_balance_weight or 0) > 0 and forward_lb_loss is not None:
                            lb_loss = forward_lb_loss.mean()
                            policy_loss = policy_loss + ctx.load_balance_weight * lb_loss
                            per_expert_losses[expert_name]["load_balance"].append(lb_loss.item())

                        # --- Offline BC (masked MSE from pre-collected dataset) ---
                        # BC obs is flat env obs → replicate across history_len
                        # to match LatticePolicy input format.
                        if ctx.bc_imitation_weight > 0:
                            with prof.section("bc_loss"):
                                bc_obs_flat, bc_target = ctx.wrapper.sample_train_bc_batch(
                                    self.device,
                                )
                                # (B, obs_dim) → (B, obs_dim, history_len)
                                bc_obs = bc_obs_flat.unsqueeze(-1).expand(
                                    -1, -1, self.history_len,
                                )
                                bc_pred_mean, _, _ = self.policy_module.get_action(
                                    bc_obs,
                                    ed["obs_signatures"],
                                    ed["action_signatures"],
                                    expert_name=expert_name,
                                    deterministic=True,
                                    return_std=False,
                                    return_value=False,
                                )
                                bc_loss = ctx.wrapper.compute_bc_loss(
                                    bc_pred_mean, bc_target, self.device,
                                )
                            policy_loss = policy_loss + ctx.bc_imitation_weight * bc_loss
                            per_expert_losses[expert_name]["bc"].append(bc_loss.item())

                        # Apply expert loss scale and accumulation scaling
                        scale = ctx.loss_scale / self.grad_accum_steps
                        policy_loss = policy_loss * scale
                        value_loss_total = value_loss_total * scale

                        with prof.section("loss.backward"):
                            (policy_loss + value_loss_total).backward()

                accum_step += 1
                is_last = sched_idx == len(schedule) - 1
                if accum_step % self.grad_accum_steps == 0 or is_last:
                    with prof.section("grad_clip + optimizer.step"):
                        if self.grad_clip > 0:
                            policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.policy_module.policy_parameters(), self.grad_clip,
                            )
                            per_expert_diag[expert_name]["grad_norms"].append(policy_grad_norm.item())
                        else:
                            total_norm = sum(
                                p.grad.norm().item() ** 2
                                for p in self.policy_module.policy_parameters() if p.grad is not None
                            ) ** 0.5
                            per_expert_diag[expert_name]["grad_norms"].append(total_norm)

                        if self.value_grad_clip > 0:
                            value_grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.policy_module.value_parameters(), self.value_grad_clip,
                            )
                            per_expert_diag[expert_name]["value_grad_norms"].append(value_grad_norm.item())

                        self.policy_optimizer.step()
                        self.value_optimizer.step()
                        self.policy_optimizer.zero_grad()
                        self.value_optimizer.zero_grad()

                pbar.update(1)

                # Сохраняем отчёт — напечатается в log_train
                if _profile_this:
                    self._mem_profile_report = prof.report()
                    prof.mem_enabled = False
                    _mem_prof_done = True

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
                diag_out = self.policy(
                    ed["states"][diag_idx],
                    ed["obs_signatures"],
                    ed["action_signatures"],
                    expert_name=expert_name,
                )
                diag_actions, diag_cov_factor, diag_diag_std, diag_values, diag_latent_std = diag_out[:5]

                logstd_mean = diag_diag_std.log().mean().item()
                logstd_min = diag_diag_std.log().min().item()
                logstd_max = diag_diag_std.log().max().item()
                act_mean = diag_actions.mean().item()
                act_std = diag_actions.std().item()
                act_abs_mean = diag_actions.abs().mean().item()
                act_min = diag_actions.min().item()
                act_max = diag_actions.max().item()
                val_post_mean = diag_values.mean().item() if diag_values is not None else 0.0
                val_post_std = diag_values.std().item() if diag_values is not None else 0.0
                latent_std_mean = diag_latent_std.mean().item() if diag_latent_std is not None else 0.0
                latent_std_min = diag_latent_std.min().item() if diag_latent_std is not None else 0.0
                latent_std_max = diag_latent_std.max().item() if diag_latent_std is not None else 0.0

                if diag_cov_factor is not None:
                    lr_var = diag_cov_factor.pow(2).sum(dim=-1).mean().item()
                    diag_var = diag_diag_std.pow(2).mean().item()
                    cov_factor_norm = diag_cov_factor.norm(dim=-1).mean().item()
                    lr_fraction = lr_var / (lr_var + diag_var + 1e-8)
                    cov_factor_abs_max = diag_cov_factor.abs().max().item()

                # MoE gate diagnostics
                gate_stats = None
                if hasattr(self.policy_module, 'get_gate_stats'):
                    obs_norm = self.policy_module.obs_normalizer(
                        ed["obs_signatures"], ed["states"][diag_idx],
                    )
                    x_diag = obs_norm[:, :, -1]
                    gate_stats = self.policy_module.get_gate_stats(x_diag)

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
                "imitation_loss_per_sample": float(np.mean(losses["imitation_per_sample"])) if losses["imitation_per_sample"] else 0.0,
                "value_loss": float(np.mean(losses["value"])) if losses["value"] else 0.0,
                "entropy_loss": float(np.mean(losses["entropy"])) if losses["entropy"] else 0.0,
                "sigma_loss": float(np.mean(losses["sigma"])) if losses["sigma"] else 0.0,
                "load_balance_loss": float(np.mean(losses["load_balance"])) if losses["load_balance"] else 0.0,
                "bc_loss": float(np.mean(losses["bc"])) if losses["bc"] else 0.0,
                "approx_kl": float(np.mean(diag["approx_kls"])) if diag["approx_kls"] else 0.0,
                "clip_frac": float(np.mean(diag["clip_fracs"])) if diag["clip_fracs"] else 0.0,
                "value_clip_frac": float(np.mean(diag["value_clip_fracs"])) if diag["value_clip_fracs"] else 0.0,
                "ratio_mean": float(np.mean(diag["ratios"])) if diag["ratios"] else 1.0,
                "grad_norm": float(np.mean(diag["grad_norms"])) if diag["grad_norms"] else 0.0,
                "value_grad_norm": float(np.mean(diag["value_grad_norms"])) if diag["value_grad_norms"] else 0.0,
                "explained_var": explained_var,
                "adv_mean": adv_mean,
                "adv_std": adv_std,
                "ret_mean": ret_mean,
                "ret_std": ret_std,
                "val_mean": val_mean,
                "val_std": val_std,
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
                "latent_std_mean": latent_std_mean,
                "latent_std_min": latent_std_min,
                "latent_std_max": latent_std_max,
                "cov_factor_norm": cov_factor_norm,
                "lr_fraction": lr_fraction,
                "cov_factor_abs_max": cov_factor_abs_max,
            }
            if gate_stats is not None:
                d = all_diagnostics[expert_name]
                d["gate_entropy"] = gate_stats["gate_entropy"]
                d["balance_entropy"] = gate_stats["balance_entropy"]
                for i, (f, p) in enumerate(zip(
                    gate_stats["routing_fracs"].tolist(),
                    gate_stats["mean_probs"].tolist(),
                )):
                    d[f"gate_frac_{i}"] = f
                    d[f"gate_prob_{i}"] = p

            # MoE shared vs routed expert contribution diagnostics
            expert_stats = getattr(self.policy_module, "_expert_stats", None)
            if expert_stats is not None:
                d = all_diagnostics[expert_name]
                for k, v in expert_stats.items():
                    d[k] = v

        # Update normalizer stats ONCE with the full batch data.
        # Must happen before the PPO loop — stats are frozen during updates
        # to avoid drift between new_log_probs and fixed_log_probs.
        with torch.no_grad():
            for expert_name, ed in expert_data.items():
                self.policy_module.obs_normalizer.update(
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

        try:
            for epoch in range(self.epoch, self.max_epochs):
                self.epoch = epoch

                # Pre-epoch: resample motions (persistent workers handle it)
                need_resample = epoch > 0 and epoch % self.resampling_interval == 0
                if self._persistent_state is not None:
                    self._persistent_state["resample_flag"].value = 1 if need_resample else 0
                elif need_resample:
                    for expert_name, ctx in self.experts.items():
                        if hasattr(ctx.wrapper.env, 'sample_motions'):
                            ctx.wrapper.env.sample_motions()

                # Sample
                t_sample_start = time.time()
                expert_batches, expert_loggers = self.sample()
                t_sample = time.time() - t_sample_start

                # Update
                t_update_start = time.time()
                all_diagnostics = self.update_params(expert_batches, debug_memory=(epoch == 0))
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
                        load_balance_loss=diag["load_balance_loss"],
                    )
                    total_steps_epoch += obc_lg.num_steps

                self.num_steps += total_steps_epoch

                if self.policy_scheduler is not None:
                    self.policy_scheduler.step()
                if self.value_scheduler is not None:
                    self.value_scheduler.step()

                # Logging
                if epoch % self.log_frequency == 0:
                    self.log_train(epoch, expert_loggers, all_diagnostics)

                # Evaluation (valid)
                if self.eval_frequency > 0 and epoch > 0 and epoch % self.eval_frequency == 0:
                    all_eval = self.evaluate()
                    self.eval_checkpoint(epoch, all_eval)

                # Evaluation (train) — deterministic eval on training data
                if self.train_eval_frequency > 0 and epoch > 0 and epoch % self.train_eval_frequency == 0:
                    train_eval = self.evaluate_train()
                    if self.use_wandb:
                        merged = {}
                        for expert_name, metrics in train_eval.items():
                            for k, v in metrics.items():
                                merged[f"{expert_name}/{k}"] = v
                        self.wandb_logger.log_eval(epoch, merged)

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

        finally:
            self._shutdown_persistent_workers()

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

            mpjpe = metrics.get("eval/mpjpe", float('inf'))
            if mpjpe < self.best_eval_mpjpe[expert_name]:
                self.best_eval_mpjpe[expert_name] = mpjpe
                self.save_checkpoint(suffix=f"best_mpjpe_{expert_name}")

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
        logger.info(f"  policy: {self.cfg.learning.policy}")
        for expert_name, ctx in self.experts.items():
            logger.info(
                f"  {expert_name}: mode={ctx.training_mode.upper()}  "
                f"ppo={ctx.ppo_weight}  obc_im={ctx.obc_imitation_weight}  "
                f"bc_im={ctx.bc_imitation_weight}  "
                f"ent={ctx.entropy_weight}  "
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
                    f"val_clip_frac={d.get('value_clip_frac', 0):.3f}  "
                    f"ratio={d.get('ratio_mean', 1):.4f}  "
                    f"grad_norm={d.get('grad_norm', 0):.3f}  "
                    f"val_grad_norm={d.get('value_grad_norm', 0):.3f}  "
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
                logger.info(
                    f"  [{expert_name}] Covariance: "
                    f"sigma_loss={d.get('sigma_loss', 0):.4f}  "
                    f"latent_std={d.get('latent_std_mean', 0):.4f} [{d.get('latent_std_min', 0):.4f}, {d.get('latent_std_max', 0):.4f}]  "
                    f"cov_norm={d.get('cov_factor_norm', 0):.4f}  "
                    f"lr_frac={d.get('lr_fraction', 0):.3f}  "
                    f"cov_max={d.get('cov_factor_abs_max', 0):.4f}"
                )

                # Imitation diagnostics
                l_im = d.get('imitation_loss', 0)
                if l_im > 0:
                    logger.info(
                        f"  [{expert_name}] OBC Imitation: "
                        f"L_im={l_im:.4f}  "
                        f"im_contrib={ctx.obc_imitation_weight * l_im:.4f}"
                    )

                # BC diagnostics
                l_bc = d.get('bc_loss', 0)
                if l_bc > 0:
                    logger.info(
                        f"  [{expert_name}] BC: "
                        f"train_loss={l_bc:.4f}  "
                        f"bc_contrib={ctx.bc_imitation_weight * l_bc:.4f}"
                    )
                if "gate_entropy" in d:
                    n_experts = self.policy_module.num_experts if hasattr(self.policy_module, 'num_experts') else 0
                    fracs = " ".join(f"{d.get(f'gate_frac_{i}', 0):.3f}" for i in range(n_experts))
                    probs = " ".join(f"{d.get(f'gate_prob_{i}', 0):.3f}" for i in range(n_experts))
                    logger.info(
                        f"  [{expert_name}] Gate: "
                        f"entropy={d['gate_entropy']:.3f}  "
                        f"balance_H={d['balance_entropy']:.3f}  "
                        f"routing=[{fracs}]  "
                        f"probs=[{probs}]"
                    )

                if "shared_mean_frac" in d:
                    alpha = getattr(self.policy_module, "alpha", 0.3)
                    parts = [
                        f"  [{expert_name}] Mean: "
                        f"shared={d['shared_mean_norm']:.4f}  "
                        f"routed={d['routed_mean_norm']:.4f}  "
                        f"shared_frac={d['shared_mean_frac']:.3f}"
                    ]
                    if "shared_cov_frac" in d:
                        parts.append(
                            f"  [{expert_name}] Cov:  "
                            f"shared={d['shared_cov_norm']:.4f}  "
                            f"routed={d['routed_cov_norm']:.4f}  "
                            f"shared_frac={d['shared_cov_frac']:.3f}"
                        )
                    if "lb_ind_entropy" in d:
                        parts.append(
                            f"  [{expert_name}] LB:   "
                            f"H_ind={d['lb_ind_entropy']:.4f}  "
                            f"H_batch={d['lb_batch_entropy']:.4f}  "
                            f"l_ind={d['lb_l_ind']:.4f}  "
                            f"l_batch={d['lb_l_batch']:.4f}  "
                            f"total={d['lb_total']:.4f}"
                        )
                    parts.append(f"  (α={alpha})")
                    logger.info("\n".join(parts))

        # Per-expert sampling profiler — только эпоха 0
        if epoch == 0 and self.profiler_reports:
            for expert_name, profiler in self.profiler_reports.items():
                logger.info(self._section_header(
                    f"Sampling Profile  |  {expert_name}"
                ))
                logger.info(profiler.report())

        # GPU Memory Profile — собран в update_params epoch 0
        if self._mem_profile_report is not None:
            logger.info(self._mem_profile_report)
            self._mem_profile_report = None

        # Per-body termination stats
        for expert_name in expert_loggers:
            ctx = self.experts[expert_name]
            env = ctx.wrapper.env
            if hasattr(env, "get_termination_stats"):
                stats = env.get_termination_stats()
                if stats and sum(s["caused"] for s in stats.values()) > 0:
                    total = sum(s["caused"] for s in stats.values())
                    parts = []
                    for body, s in sorted(stats.items(), key=lambda x: -x[1]["caused"]):
                        if s["caused"] > 0:
                            parts.append(f"{body}={s['caused']}({100 * s['caused'] / total:.0f}% d={s['mean_dist'] * 100:.1f}cm)")
                    logger.info(f"  [{expert_name}] Terminations ({total}): {' | '.join(parts)}")

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

            if self.vectorized_eval and self.device.type in ('cuda', 'mps'):
                all_eval[expert_name] = self._evaluate_expert_vectorized(
                    expert_name, ctx, valid_wrapper,
                )
            else:
                all_eval[expert_name] = self.evaluate_expert(
                    expert_name, ctx, valid_wrapper,
                )

            # BC test loss
            if ctx.bc_imitation_weight > 0 and hasattr(ctx.wrapper, '_bc_test') and ctx.wrapper._bc_test is not None:
                bc_test_losses = []
                with torch.no_grad():
                    for t_obs, t_act in ctx.wrapper.iter_test_bc_batches(
                        self.batch_size, self.device,
                    ):
                        t_obs_ts = t_obs.unsqueeze(-1).expand(-1, -1, self.history_len)
                        t_pred, _, _ = self.policy_module.get_action(
                            t_obs_ts,
                            ctx.parser.obs_signatures,
                            ctx.action_parser.action_signatures,
                            expert_name=expert_name,
                            deterministic=True,
                            return_std=False,
                            return_value=False,
                        )
                        bl = ctx.wrapper.compute_bc_loss(t_pred, t_act, self.device)
                        bc_test_losses.append(bl.item())
                bc_test_loss = float(np.mean(bc_test_losses))
                all_eval[expert_name]["eval/bc_test_loss"] = bc_test_loss
                logger.info(f"  [{expert_name}] BC test_loss={bc_test_loss:.4f}")

        return all_eval

    def evaluate_train(self) -> Dict[str, Dict[str, float]]:
        """
        Deterministic eval на train выборке (без exploration noise).
        Логирует с prefix train_eval/ для сравнения с eval/.
        """
        logger.info(self._section_header("Train Evaluation"))

        # Temporarily swap valid_experts to train wrappers
        saved_valid = self.valid_experts
        self.valid_experts = {n: ctx.wrapper for n, ctx in self.experts.items()}
        for ctx in self.experts.values():
            ctx.wrapper.env.start_eval(im_eval=True)

        all_eval: Dict[str, Dict[str, float]] = {}
        try:
            for expert_name, ctx in self.experts.items():
                wrapper = ctx.wrapper
                if self.vectorized_eval and self.device.type in ('cuda', 'mps'):
                    metrics = self._evaluate_expert_vectorized(
                        expert_name, ctx, wrapper,
                    )
                else:
                    metrics = self.evaluate_expert(
                        expert_name, ctx, wrapper,
                    )
                # Rename eval/ → train_eval/
                all_eval[expert_name] = {
                    k.replace("eval/", "train_eval/"): v
                    for k, v in metrics.items()
                }
        finally:
            for ctx in self.experts.values():
                ctx.wrapper.env.end_eval()
            self.valid_experts = saved_valid

        return all_eval

    @staticmethod
    def _build_eval_metrics(
        expert_name: str,
        episode_rewards: list,
        episode_lengths: list,
        episode_imitation_losses: list,
        episode_mpjpes: list,
        episode_frame_coverages: list,
        episode_successes: list,
        use_expert: bool,
        label: str = "",
    ) -> Dict[str, float]:
        """Build metrics dict and log summary. Shared by sequential and vectorized eval."""
        metrics = {
            "eval/mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "eval/std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
            "eval/mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "eval/std_length": float(np.std(episode_lengths)) if episode_lengths else 0.0,
        }
        if episode_imitation_losses:
            metrics["eval/imitation_loss"] = float(np.mean(episode_imitation_losses))
        if episode_mpjpes:
            metrics["eval/mpjpe"] = float(np.mean(episode_mpjpes))
        if episode_frame_coverages:
            metrics["eval/frame_coverage"] = float(np.mean(episode_frame_coverages))
        if episode_successes:
            metrics["eval/success_rate"] = float(np.mean(episode_successes))

        extra = ""
        if "eval/mpjpe" in metrics:
            extra += f", mpjpe={metrics['eval/mpjpe'] * 1000:.2f}mm"
        if "eval/frame_coverage" in metrics:
            extra += f", frame_cov={metrics['eval/frame_coverage']:.3f}"
        if "eval/success_rate" in metrics:
            extra += f", success={metrics['eval/success_rate']:.3f}"

        tag = f" ({label})" if label else ""
        logger.info(
            f"Eval [{expert_name}]{tag}: "
            f"reward={metrics['eval/mean_reward']:.4f}±{metrics['eval/std_reward']:.4f}, "
            f"length={metrics['eval/mean_length']:.2f}±{metrics['eval/std_length']:.2f}"
            + (f", im_loss={metrics.get('eval/imitation_loss', 0):.4f}" if use_expert else "")
            + extra
        )
        return metrics

    def evaluate_expert(
        self,
        expert_name: str,
        ctx: ExpertContext,
        valid_wrapper,
        return_per_motion: bool = False,
    ) -> Dict[str, float]:
        """Sequential evaluation для одного эксперта (CPU inference)."""
        self.policy.eval()
        valid_parser = ObservationParser.from_env(valid_wrapper.env, self.history_len)
        valid_action_parser = ActionParser.from_env(valid_wrapper.env)

        episode_rewards = []
        episode_lengths = []
        episode_imitation_losses = []
        episode_mpjpes = []
        episode_frame_coverages = []
        episode_successes = []
        per_motion_results: List[Dict[str, Any]] = []

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

            for t in range(10000):
                obs_ts, obs_sigs = valid_parser.get_observation(self.device)
                act_sigs = valid_action_parser.action_signatures

                with torch.no_grad():
                    action, _, _ = self.policy_module.get_action(
                        obs_ts, obs_sigs, act_sigs,
                        expert_name=expert_name,
                        deterministic=True,
                        return_value=False,
                    )

                    if ctx.use_expert:
                        expert_action = valid_wrapper.get_expert_action(obs)
                        expert_action_t = torch.from_numpy(expert_action).float().to(self.device)
                        im_loss = ((action.squeeze(0) - expert_action_t) ** 2).mean().item()
                        step_imitation_losses.append(im_loss)

                action_np = action.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = valid_wrapper.step(action_np)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                valid_parser.update(next_obs)
                obs = next_obs

                if done:
                    if "mpjpe" in info:
                        episode_mpjpes.append(float(info["mpjpe"]))
                    if "frame_coverage" in info:
                        episode_frame_coverages.append(float(info["frame_coverage"]))
                    if "success" in info:
                        episode_successes.append(float(info["success"]))
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if step_imitation_losses:
                episode_imitation_losses.append(np.mean(step_imitation_losses))

            if return_per_motion:
                per_motion_results.append({
                    "motion_id": motion_id,
                    "reward": episode_reward,
                    "length": episode_length,
                    "im_loss": float(np.mean(step_imitation_losses)) if step_imitation_losses else None,
                    "mpjpe": info.get("mpjpe"),
                    "max_mpjpe": info.get("max_mpjpe"),
                    "frame_coverage": info.get("frame_coverage"),
                    "success": info.get("success"),
                })

        metrics = self._build_eval_metrics(
            expert_name, episode_rewards, episode_lengths,
            episode_imitation_losses, episode_mpjpes,
            episode_frame_coverages, episode_successes,
            use_expert=ctx.use_expert,
        )

        if return_per_motion:
            return metrics, per_motion_results
        return metrics

    # ------------------------------------------------------------------
    #  Vectorized Evaluation
    # ------------------------------------------------------------------

    def _eval_env_worker(
        self,
        expert_name: str,
        worker_id: int,
        obs_buf: torch.Tensor,
        action_buf: torch.Tensor,
        expert_action_buf: torch.Tensor,
        obs_ready,
        action_ready,
        stop_flag,
        motion_queue,
        result_queue,
        send_sigs: bool,
    ) -> None:
        """
        Eval env worker. Pulls motion_ids from motion_queue, runs episodes,
        reports per-episode metrics via result_queue.
        """
        self.seed_worker(worker_id)
        ctx = self.experts[expert_name]
        wrapper = self.valid_experts[expert_name]
        env = wrapper.env
        parser = ObservationParser.from_env(env, self.history_len)
        action_parser = ActionParser.from_env(env)

        env.start_eval(im_eval=True)

        def _load_next_motion() -> bool:
            """Pull next motion from queue, reset env. Returns False if no more."""
            try:
                motion_id = motion_queue.get_nowait()
            except Exception:
                return False
            env._current_eval_motion_id = int(motion_id)
            env._active_motion_ids = np.array([motion_id])
            return True

        eval_uses_wrapper_expert = ctx.use_expert

        def _reset_and_fill_obs():
            """Reset env, fill shared obs buffer."""
            nonlocal obs, send_sigs
            obs, info = wrapper.reset()
            parser.reset(obs)
            obs_ts, obs_sigs = parser.get_observation(torch.device("cpu"))
            act_sigs = action_parser.action_signatures
            if send_sigs:
                result_queue.put(("sigs", expert_name, obs_sigs, act_sigs))
                send_sigs = False
            obs_buf.copy_(obs_ts.squeeze(0))
            if eval_uses_wrapper_expert:
                ea = wrapper.get_expert_action(obs)
                expert_action_buf.copy_(torch.from_numpy(ea).float())

        obs = None

        try:
            # Load first motion
            if not _load_next_motion():
                return
            _reset_and_fill_obs()
            obs_ready.set()

            ep_reward = 0.0
            ep_length = 0
            ep_im_losses = []

            while True:
                action_ready.wait()
                action_ready.clear()

                if stop_flag.value:
                    env.end_eval()
                    return

                action_np = action_buf.numpy().copy()

                if eval_uses_wrapper_expert:
                    ea_np = expert_action_buf.numpy().copy()
                    ep_im_losses.append(float(((action_np - ea_np) ** 2).mean()))

                next_obs, reward, terminated, truncated, info = wrapper.step(action_np)
                done = terminated or truncated
                ep_reward += reward
                ep_length += 1

                if done:
                    ep_result = {
                        "motion_id": env._current_eval_motion_id,
                        "reward": ep_reward,
                        "length": ep_length,
                        "im_loss": float(np.mean(ep_im_losses)) if ep_im_losses else None,
                        "mpjpe": info.get("mpjpe"),
                        "max_mpjpe": info.get("max_mpjpe"),
                        "frame_coverage": info.get("frame_coverage"),
                        "success": info.get("success"),
                    }
                    if terminated:
                        ep_result["termination_body"] = info.get("termination_body")
                    result_queue.put(("episode", expert_name, ep_result))

                    # Try next motion
                    if not _load_next_motion():
                        break
                    _reset_and_fill_obs()
                    ep_reward = 0.0
                    ep_length = 0
                    ep_im_losses = []
                    obs_ready.set()
                    continue

                parser.update(next_obs)
                obs = next_obs
                obs_ts, _ = parser.get_observation(torch.device("cpu"))
                obs_buf.copy_(obs_ts.squeeze(0))
                if eval_uses_wrapper_expert:
                    ea = wrapper.get_expert_action(obs)
                    expert_action_buf.copy_(torch.from_numpy(ea).float())
                obs_ready.set()

        except Exception as e:
            import traceback
            print(f"EvalWorker {worker_id} [{expert_name}] failed: {e}")
            traceback.print_exc()

        finally:
            env.end_eval()
            result_queue.put(("done", expert_name, worker_id))

    def _evaluate_expert_vectorized(
        self,
        expert_name: str,
        ctx: ExpertContext,
        valid_wrapper,
        return_per_motion: bool = False,
    ) -> Dict[str, float]:
        """
        Vectorized evaluation: N env workers step on CPU,
        main process does batched inference on GPU/MPS.

        If return_per_motion=True, returns a tuple (metrics, per_motion_results)
        where per_motion_results is a list of dicts with per-motion metrics.
        """
        self.policy.eval()

        n_workers = ctx.vec_batch_size or ctx.num_threads
        total_motions = valid_wrapper.num_motions
        # Don't spawn more workers than motions
        n_workers = min(n_workers, total_motions)

        # Fill motion queue
        motion_queue = fork_ctx.Queue()
        for motion_id in valid_wrapper.env._all_motion_ids:
            motion_queue.put(int(motion_id))

        result_queue = fork_ctx.Queue()
        stop_flag = fork_ctx.Value('i', 0)

        # Spawn workers
        obs_shape = (ctx.parser.n_obs_elements, self.history_len)
        action_dim = ctx.wrapper.action_dim
        device = self.device
        device_type = "cuda" if device.type == "cuda" else "cpu"
        use_amp = self.use_bfloat16 and device.type == "cuda"

        class _WH:
            __slots__ = (
                'obs_buf', 'action_buf', 'expert_action_buf',
                'obs_ready', 'action_ready', 'proc', 'active',
            )

        workers: List[_WH] = []
        for w_idx in range(n_workers):
            wh = _WH()
            wh.obs_buf = torch.zeros(obs_shape, dtype=torch.float32).share_memory_()
            wh.action_buf = torch.zeros(action_dim, dtype=torch.float32).share_memory_()
            wh.expert_action_buf = torch.zeros(action_dim, dtype=torch.float32).share_memory_()
            wh.obs_ready = fork_ctx.Event()
            wh.action_ready = fork_ctx.Event()
            wh.active = True
            wh.proc = fork_ctx.Process(
                target=self._eval_env_worker,
                args=(
                    expert_name, w_idx + 1,
                    wh.obs_buf, wh.action_buf, wh.expert_action_buf,
                    wh.obs_ready, wh.action_ready,
                    stop_flag, motion_queue, result_queue,
                    w_idx == 0,  # send_sigs
                ),
                daemon=True,
            )
            workers.append(wh)

        for wh in workers:
            wh.proc.start()

        # Collect sigs
        obs_sigs, act_sigs = None, None
        msg = result_queue.get(timeout=30)
        if msg[0] == "sigs":
            obs_sigs, act_sigs = msg[2], msg[3]

        # Episode results
        episode_rewards = []
        episode_lengths = []
        episode_imitation_losses = []
        episode_mpjpes = []
        episode_frame_coverages = []
        episode_successes = []
        per_motion_results: List[Dict[str, Any]] = []

        pbar = tqdm(total=total_motions, desc=f"Eval [{expert_name}]", unit="ep")
        workers_done = 0

        # Main inference loop
        with torch.no_grad():
            while workers_done < n_workers:
                # Drain result_queue for completed episodes / done workers
                while True:
                    try:
                        msg = result_queue.get_nowait()
                        if msg[0] == "episode":
                            m = msg[2]
                            episode_rewards.append(m["reward"])
                            episode_lengths.append(m["length"])
                            if m["im_loss"] is not None:
                                episode_imitation_losses.append(m["im_loss"])
                            if m["mpjpe"] is not None:
                                episode_mpjpes.append(float(m["mpjpe"]))
                            if m["frame_coverage"] is not None:
                                episode_frame_coverages.append(float(m["frame_coverage"]))
                            if m["success"] is not None:
                                episode_successes.append(float(m["success"]))
                            if return_per_motion:
                                per_motion_results.append(m)
                            pbar.update(1)
                        elif msg[0] == "done":
                            workers_done += 1
                        elif msg[0] == "sigs":
                            obs_sigs, act_sigs = msg[2], msg[3]
                    except Exception:
                        break

                if workers_done >= n_workers:
                    break

                # Wait for active workers' obs
                active_workers = []
                for wh in workers:
                    if not wh.active:
                        continue
                    got_obs = wh.obs_ready.wait(timeout=0.5)
                    if got_obs:
                        wh.obs_ready.clear()
                        active_workers.append(wh)
                    elif not wh.proc.is_alive():
                        wh.active = False

                if not active_workers:
                    continue

                # Batched GPU inference
                batch_obs = torch.stack([wh.obs_buf.clone() for wh in active_workers])

                with torch.autocast(
                    device_type=device_type,
                    dtype=torch.bfloat16,
                    enabled=use_amp,
                ):
                    actions, _, _ = self.policy_module.get_action(
                        batch_obs.to(device),
                        obs_sigs, act_sigs,
                        expert_name=expert_name,
                        deterministic=True,
                        return_std=False,
                        return_value=False,
                    )
                actions_cpu = actions.cpu()

                for i, wh in enumerate(active_workers):
                    wh.action_buf.copy_(actions_cpu[i])
                    if ctx.use_expert:
                        # expert_action already in buf from worker
                        pass
                    wh.action_ready.set()

        pbar.close()

        # Cleanup: drain remaining messages, then terminate
        stop_flag.value = 1
        for wh in workers:
            wh.action_ready.set()

        # Drain any remaining messages (workers may block on put if queue is full)
        deadline = time.time() + 10
        while workers_done < n_workers and time.time() < deadline:
            try:
                msg = result_queue.get(timeout=0.5)
                if msg[0] == "done":
                    workers_done += 1
                elif msg[0] == "episode" and return_per_motion:
                    per_motion_results.append(msg[2])
            except Exception:
                pass

        for wh in workers:
            wh.proc.join(timeout=3)
            if wh.proc.is_alive():
                wh.proc.terminate()

        metrics = self._build_eval_metrics(
            expert_name, episode_rewards, episode_lengths,
            episode_imitation_losses, episode_mpjpes,
            episode_frame_coverages, episode_successes,
            use_expert=ctx.use_expert,
            label=f"vectorized, {n_workers} workers",
        )

        if return_per_motion:
            return metrics, per_motion_results
        return metrics

    # ------------------------------------------------------------------
    #  Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, suffix: str = "latest") -> None:
        expert_modes = {n: ctx.training_mode for n, ctx in self.experts.items()}
        checkpoint = {
            "epoch": self.epoch,
            "num_steps": self.num_steps,
            "policy": self.policy_module.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "expert_names": list(self.experts.keys()),
            "expert_modes": expert_modes,
        }
        if self.policy_scheduler is not None:
            checkpoint["policy_scheduler"] = self.policy_scheduler.state_dict()
        if self.value_scheduler is not None:
            checkpoint["value_scheduler"] = self.value_scheduler.state_dict()

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

        # DataParallel: убираем префикс "module." в checkpoint если есть
        policy_state = checkpoint["policy"]
        if policy_state and next(iter(policy_state.keys()), "").startswith("module."):
            policy_state = {k.replace("module.", "", 1): v for k, v in policy_state.items()}

        try:
            self.policy_module.load_state_dict(policy_state)
        except RuntimeError as e:
            logger.warning(f"Strict load failed. Exception: {e}")
            logger.info("Filtering checkpoint for size mismatches (cross-environment transfer)...")

            model_state = self.policy_module.state_dict()
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

            self.policy_module.load_state_dict(filtered_state, strict=False)

        if resume_training:
            self.epoch = checkpoint["epoch"] + 1
            self.num_steps = checkpoint["num_steps"]
            if "policy_optimizer" in checkpoint:
                self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
            elif "optimizer" in checkpoint:
                logger.warning("Loading legacy single-optimizer checkpoint into policy_optimizer")
                self.policy_optimizer.load_state_dict(checkpoint["optimizer"])
            if "value_optimizer" in checkpoint:
                self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
            if self.policy_scheduler is not None and "policy_scheduler" in checkpoint:
                self.policy_scheduler.load_state_dict(checkpoint["policy_scheduler"])
            if self.value_scheduler is not None and "value_scheduler" in checkpoint:
                self.value_scheduler.load_state_dict(checkpoint["value_scheduler"])
        else:
            self.epoch = 0
            self.num_steps = 0

            # Normalizer stats keyed by semantic signature (e.g. "femur|r|position|x"),
            # so stats from source expert transfer naturally to target expert
            # for shared body parts. Only prune stats not needed by any new expert.
            ckpt_keys = set(self.policy_module.obs_normalizer.stats.keys())
            needed_keys = set()
            for ctx in self.experts.values():
                for sig in ctx.parser.obs_signatures:
                    needed_keys.add(self.policy_module.obs_normalizer._sig_key(sig))

            reused = ckpt_keys & needed_keys
            pruned = ckpt_keys - needed_keys
            fresh = needed_keys - ckpt_keys

            for key in pruned:
                del self.policy_module.obs_normalizer.stats[key]
            self.policy_module.obs_normalizer._invalidate_cache()

            logger.info(
                f"Transfer mode: normalizer stats — "
                f"{len(reused)} reused, {len(fresh)} new (no stats yet), "
                f"{len(pruned)} pruned (not in target experts)"
            )

            if self.cfg.learning.get("transfer_keep_value", False):
                logger.info("Transfer mode: keeping value net from checkpoint")
            else:
                logger.info("Transfer mode: reinitializing value nets")
                if self.cfg.learning.policy == "transformer":
                    for module in [self.policy_module.value_decoder, self.policy_module.value_head]:
                        for param in module.parameters():
                            if param.dim() >= 2:
                                nn.init.xavier_uniform_(param)
                            elif param.dim() == 1:
                                nn.init.zeros_(param)
                    nn.init.normal_(self.policy_module.value_query.data)
                elif self.cfg.learning.policy in ("lattice", "moe"):
                    for param in self.policy_module.value_net.parameters():
                        if param.dim() >= 2:
                            nn.init.xavier_uniform_(param)
                        elif param.dim() == 1:
                            nn.init.zeros_(param)
                    self.policy_module.value_head.weight.data.mul_(0.0)
                    nn.init.xavier_uniform_(self.policy_module.value_head.weight)
                    self.policy_module.value_head.weight.data.mul_(0.1)
                    self.policy_module.value_head.bias.data.zero_()

            # Reinit action head for masked indices (e.g. arms after OBC on legs-only teacher)
            self._transfer_reinit_masked_actions()

        prev_mode = checkpoint.get("training_mode", "unknown")
        src_epoch = checkpoint.get("epoch", "?")
        logger.info(
            f"Loaded checkpoint from {path} "
            f"(epoch={src_epoch}, mode={prev_mode}), "
            f"resuming from epoch {self.epoch}"
        )

    def _transfer_reinit_masked_actions(self) -> None:
        """Reinit action head rows for masked action indices (e.g. arms after OBC on legs-only teacher).

        Uses get_action_mask() from any expert wrapper that provides it.
        Indices where mask=0 get reinitialized in:
          - action_mean.weight (rows)
          - action_mean.bias
          - log_std (diagonal part)
        """
        if self.cfg.learning.policy != "lattice":
            raise NotImplementedError(
                "Transfer reinit for masked actions not implemented for this policy"
            )

        # Collect union of masked indices across all experts
        masked_indices = set()
        for ctx in self.experts.values():
            if hasattr(ctx.wrapper, 'get_action_mask'):
                mask = ctx.wrapper.get_action_mask()
                zero_idx = (mask == 0).nonzero(as_tuple=True)[0].tolist()
                masked_indices.update(zero_idx)

        if not masked_indices:
            return

        idx = sorted(masked_indices)
        idx_t = torch.tensor(idx, dtype=torch.long)

        pm = self.policy_module
        with torch.no_grad():
            # action_mean: nn.Linear(latent_dim, action_dim)
            # weight shape: (action_dim, latent_dim) — reinit rows
            fan_in = pm.action_mean.weight.shape[1]
            std = (2.0 / (fan_in + 1)) ** 0.5
            pm.action_mean.weight[idx_t] = torch.randn_like(
                pm.action_mean.weight[idx_t],
            ) * std * 0.1
            pm.action_mean.bias[idx_t] = 0.0

            # log_std: (1, action_dim + latent_dim)
            sub_cfg = self.cfg.learning.get("lattice", {})
            log_std_init = sub_cfg.get("log_std_init", 0.0) if sub_cfg else 0.0
            pm.log_std.data[0, idx_t] = log_std_init

        logger.info(
            f"Transfer mode: reinitialized action head for {len(idx)} masked indices "
            f"(arms: {idx[0]}..{idx[-1]})"
        )
