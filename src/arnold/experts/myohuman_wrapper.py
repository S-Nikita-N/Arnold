"""
Обёртка для среды MyoHuman (full-body musculoskeletal model).

Аналогична KinesisWrapper, но работает с полнотелой моделью myohuman.xml.
Поддерживает два режима:
  1. С экспертом — загружает обученную Lattice/Gaussian policy (для OBC/OBC-PPO)
  2. Без эксперта — только среда (для чистого PPO)
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch

from typing import Tuple, Iterator
from omegaconf import DictConfig
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


# Пути к подмодулю Myohuman
MYOHUMAN_ROOT = Path(__file__).parent / "Myohuman"
MYOHUMAN_SRC = MYOHUMAN_ROOT / "src"
MYOHUMAN_CFG = MYOHUMAN_ROOT / "cfg"
MYOHUMAN_DATA = MYOHUMAN_ROOT / "data"

# Добавляем в sys.path для корректных импортов
if str(MYOHUMAN_ROOT) not in sys.path:
    sys.path.insert(0, str(MYOHUMAN_ROOT))
if str(MYOHUMAN_SRC) not in sys.path:
    sys.path.insert(0, str(MYOHUMAN_SRC))


logger = logging.getLogger(__name__)


def load_myohuman_config(
    config_dir: str = None,
    overrides: list = None,
) -> DictConfig:
    """
    Загружает полный конфиг MyoHuman через Hydra.

    Args:
        config_dir: Путь к директории cfg MyoHuman (по умолчанию — из submodule)
        overrides: Список Hydra overrides

    Returns:
        DictConfig с полным конфигом
    """
    if config_dir is None:
        config_dir = str(MYOHUMAN_CFG)

    if overrides is None:
        overrides = []

    # Очищаем предыдущую инициализацию Hydra если есть
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    return cfg


class MyoHumanWrapper:
    """
    Интерфейс к среде MyoHuman (полнотелая модель).

    Два режима работы:
    - С экспертом: загружает обученную policy из checkpoint (для OBC/OBC-PPO)
    - Без эксперта: только среда (для PPO), get_expert_action() недоступен

    Интерфейс:
    - reset() → (obs, info)
    - step(action) → (obs, reward, terminated, truncated, info)
    - get_expert_action(flat_obs) → np.ndarray
    - forward_motions() → Iterator[int]
    - env property → среда
    """

    def __init__(
        self,
        cfg_path: str = None,
        expert_cfg: DictConfig = None,
        checkpoint_epoch: int = 0,
        device: str = "cpu",
        overrides: list = [],
        mode: str = "train",
        simple: bool = False,
    ):
        """
        Args:
            cfg_path: Путь к директории cfg MyoHuman (None для default)
            expert_cfg: Готовый DictConfig (для multiprocessing, вместо cfg_path)
            checkpoint_epoch: Эпоха чекпоинта эксперта (0 = без эксперта, -1 = latest)
            device: Устройство ("cpu" или "cuda")
            overrides: Hydra overrides
            mode: "train" или "valid" — определяет motion file и настройки
            simple: Использовать упрощённую модель (simpletorso)
        """
        self.mode = mode
        self.simple = simple
        self.device = torch.device(device)
        self._expert_policy = None

        original_cwd = os.getcwd()
        os.chdir(MYOHUMAN_ROOT)

        try:
            from myohuman.env.myolegs_im import MyoLegsIm

            if expert_cfg is not None:
                self.cfg = expert_cfg
            else:
                if mode == "valid":
                    run_config = "run=eval_run"
                else:
                    run_config = "run=train_run"

                default_overrides = [
                    run_config,
                    "no_log=True",
                    "test=True" if mode == "valid" else "test=False",
                ]

                if simple:
                    xml_path = MYOHUMAN_ROOT / "xml" / "myohuman_simpletorso.xml"
                    pose_file = "ik_test_simpletorso.pkl" if mode == "valid" else "ik_train_simpletorso.pkl"
                    pose_path = MYOHUMAN_ROOT / "data" / "inverse_kinematics" / pose_file
                    default_overrides.append(f"run.xml_path={xml_path}")
                    default_overrides.append(f"run.initial_pose_file={pose_path}")

                if overrides:
                    default_overrides.extend(overrides)

                if not any(override.startswith("run.headless=") for override in overrides):
                    default_overrides.append("run.headless=True")

                cfg_dir = cfg_path if cfg_path else str(MYOHUMAN_CFG)
                self.cfg = load_myohuman_config(
                    config_dir=cfg_dir, overrides=default_overrides
                )

            # Устанавливаем project_root на корень Myohuman
            self.cfg.project_root = str(MYOHUMAN_ROOT)

            # Создаём среду напрямую (без полного AgentIM)
            self._env = MyoLegsIm(self.cfg)

            # Параметры среды
            self.action_dim = self._env.action_space.shape[0]
            self.obs_dim = self._env.observation_space.shape[0]
            self.actions_low = self._env.action_space.low.copy()
            self.actions_high = self._env.action_space.high.copy()

            # Загружаем эксперта если указан checkpoint
            if checkpoint_epoch != 0:
                self._load_expert(checkpoint_epoch)

            # Загружаем движения (инициализируем active set)
            if not (mode == "valid" and self.cfg.run.im_eval):
                self._env.sample_motions()

        finally:
            os.chdir(original_cwd)

    def _load_expert(self, checkpoint_epoch: int) -> None:
        """
        Загружает обученную policy эксперта из checkpoint.

        Args:
            checkpoint_epoch: -1 для latest, >0 для конкретной эпохи
        """
        from myohuman.learning.policy_lattice import PolicyLattice
        from myohuman.learning.policy_gaussian import PolicyGaussian

        state_dim = self._env.observation_space.shape[0]
        action_dim = self._env.action_space.shape[0]

        # Создаём policy сеть по конфигу
        actor_type = self.cfg.learning.actor_type
        if actor_type == "lattice":
            self._expert_policy = PolicyLattice(
                self.cfg, action_dim=action_dim, latent_dim=512, state_dim=state_dim
            )
        elif actor_type == "gauss":
            self._expert_policy = PolicyGaussian(
                self.cfg, action_dim=action_dim, state_dim=state_dim
            )
        else:
            raise ValueError(f"Unknown actor_type: {actor_type}")

        # Загружаем чекпоинт
        if checkpoint_epoch == -1:
            ckpt_path = os.path.join(self.cfg.output_dir, "model.pth")
        else:
            ckpt_path = os.path.join(
                self.cfg.output_dir, f"model_epoch_{checkpoint_epoch}.pth"
            )

        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self._expert_policy.load_state_dict(state["policy"])
            self._expert_policy.to(self.device)
            self._expert_policy.eval()
            logger.info(f"MyoHuman expert loaded from {ckpt_path}")
        else:
            logger.warning(f"Expert checkpoint not found: {ckpt_path}")
            self._expert_policy = None

    @property
    def has_expert(self) -> bool:
        """Есть ли загруженный эксперт."""
        return self._expert_policy is not None

    def reset(self) -> Tuple[np.ndarray, dict]:
        """Сброс среды и возврат obs."""
        obs, info = self._env.reset()
        return obs, info

    def forward_motions(self) -> Iterator[int]:
        """
        Итератор по всем движениям в библиотеке.
        Каждая итерация загружает следующее движение.
        После yield нужно вызвать reset().

        Yields:
            int: Индекс текущего движения.
        """
        return self._env.forward_motions()

    def get_expert_action(self, flat_obs: np.ndarray) -> np.ndarray:
        """
        Получить действие эксперта по плоскому obs.

        Raises:
            RuntimeError: Если эксперт не загружен.
        """
        if self._expert_policy is None:
            raise RuntimeError(
                "Expert policy not loaded. Use checkpoint_epoch != 0 to load expert, "
                "or set obc_imitation_weight=0 for PPO-only training."
            )

        with torch.no_grad():
            obs_t = torch.from_numpy(flat_obs).to(self.device).float()
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            action = self._expert_policy.select_action(obs_t, mean_action=True)[0]
            return action.cpu().numpy().squeeze()

    def preprocess_actions(self, action: np.ndarray) -> np.ndarray:
        """
        Clip и rescale действий (аналогично KinesisWrapper).
        Среда ожидает действия в [actions_low, actions_high].
        """
        action = np.clip(action.astype(np.float32), self.actions_low, self.actions_high)
        d = (self.actions_high - self.actions_low) / 2.0
        m = (self.actions_low + self.actions_high) / 2.0
        return action * d + m

    def step(self, action: np.ndarray):
        """
        Шаг в среде с предварительным clip + rescale действий.
        """
        action = self.preprocess_actions(action)
        next_obs, reward, terminated, truncated, info = self._env.step(action)

        # Стандартизируем формат info
        if "r_body_pos" in info and isinstance(info["r_body_pos"], np.ndarray):
            info["r_body_pos"] = info["r_body_pos"][0] if info["r_body_pos"].ndim > 0 else info["r_body_pos"]
        if "r_vel" in info and isinstance(info["r_vel"], np.ndarray):
            info["r_vel"] = info["r_vel"][0] if info["r_vel"].ndim > 0 else info["r_vel"]

        return next_obs, reward, terminated, truncated, info

    @property
    def env(self):
        """Доступ к среде."""
        return self._env

    @property
    def num_motions(self) -> int:
        """Количество загруженных движений."""
        return len(self._env._all_motion_ids)

    def sample_motions(self, num_motions: int = None) -> None:
        """Пересэмплирует движения из библиотеки."""
        if hasattr(self._env, "sample_motions"):
            self._env.sample_motions()

    # ------------------------------------------------------------------
    #  Offline BC (Behavioral Cloning) from pre-collected demonstrations
    # ------------------------------------------------------------------

    def load_bc_datasets(
        self,
        dataset_dir: str,
        batch_size: int = 256,
        success_only: bool = False,
        max_mpjpe_threshold: float = None,
        mean_mpjpe_threshold: float = None,
    ) -> None:
        """Load BC datasets from *dataset_dir*.

        Expected files: train.json, train.npz, test.json, test.npz.
        JSON contains per-motion metrics + per-frame metrics used for filtering.
        NPZ contains obs, actions, action_mask, motion_ids, frame_idx.
        """
        d = Path(dataset_dir)
        self._bc_batch_size = batch_size

        train_npz = d / "train.npz"
        train_json = d / "train.json"
        test_npz = d / "test.npz"
        test_json = d / "test.json"

        if not train_npz.exists():
            raise FileNotFoundError(f"BC train dataset not found: {train_npz}")
        if not train_json.exists():
            raise FileNotFoundError(f"BC train metadata not found: {train_json}")

        logger.info("Loading BC datasets from %s", d)
        logger.info("  filters: success_only=%s  max_mpjpe=%s  mean_mpjpe=%s  batch_size=%d",
                     success_only, max_mpjpe_threshold, mean_mpjpe_threshold, batch_size)

        logger.info("[train]")
        self._bc_train = self._load_and_filter(
            train_npz, train_json,
            success_only, max_mpjpe_threshold, mean_mpjpe_threshold,
        )

        if test_npz.exists() and test_json.exists():
            logger.info("[test]")
            self._bc_test = self._load_and_filter(
                test_npz, test_json,
                success_only, max_mpjpe_threshold, mean_mpjpe_threshold,
            )
        else:
            self._bc_test = None
            logger.info("[test] not found, skipping")

    @staticmethod
    def _load_and_filter(
        npz_path: Path,
        json_path: Path,
        success_only: bool,
        max_mpjpe_threshold: float,
        mean_mpjpe_threshold: float,
    ) -> dict:
        """Load npz + json, build per-frame mask, return filtered tensors."""
        import json as json_mod

        data = np.load(str(npz_path))
        obs = data["obs"]               # (N, obs_dim)
        actions = data["actions"]        # (N, act_dim)
        action_mask = data["action_mask"]  # (act_dim,)
        motion_ids = data["motion_ids"]  # (N,)
        frame_idx = data["frame_idx"]    # (N,)

        with open(json_path) as f:
            meta = json_mod.load(f)

        # Build per-motion lookup: motion_id → result dict
        by_motion = {r["motion_id"]: r for r in meta["results"]}
        total_motions = len(by_motion)
        successful_motions = sum(1 for r in by_motion.values() if r["success"])

        # Per-frame boolean mask + per-reason counters
        n_total = len(obs)
        keep = np.ones(n_total, dtype=bool)
        dropped_success = 0
        dropped_max_mpjpe = 0
        dropped_mean_mpjpe = 0
        dropped_missing = 0

        for i in range(n_total):
            mid = int(motion_ids[i])
            fid = int(frame_idx[i])
            r = by_motion.get(mid)
            if r is None:
                keep[i] = False
                dropped_missing += 1
                continue

            if success_only and not r["success"]:
                keep[i] = False
                dropped_success += 1
                continue

            fm = r["frame_metrics"]
            if fid < len(fm):
                if max_mpjpe_threshold is not None and fm[fid]["max_mpjpe"] > max_mpjpe_threshold:
                    keep[i] = False
                    dropped_max_mpjpe += 1
                    continue
                if mean_mpjpe_threshold is not None and fm[fid]["mean_mpjpe"] > mean_mpjpe_threshold:
                    keep[i] = False
                    dropped_mean_mpjpe += 1

        obs = obs[keep]
        actions = actions[keep]
        n_kept = len(obs)
        kept_motion_ids = set(motion_ids[keep].tolist())

        if n_kept == 0:
            raise ValueError(
                f"BC filtering removed all frames from {npz_path} "
                f"(success_only={success_only}, max_mpjpe={max_mpjpe_threshold}, "
                f"mean_mpjpe={mean_mpjpe_threshold})"
            )

        logger.info("BC dataset: %s", npz_path)
        logger.info("  motions: %d total, %d successful, %d kept after filter",
                     total_motions, successful_motions, len(kept_motion_ids))
        logger.info("  frames:  %d / %d kept (%.1f%%)",
                     n_kept, n_total, 100 * n_kept / n_total)
        if dropped_success:
            logger.info("  dropped by success_only:       %d frames", dropped_success)
        if dropped_max_mpjpe:
            logger.info("  dropped by max_mpjpe > %.4f:   %d frames",
                         max_mpjpe_threshold, dropped_max_mpjpe)
        if dropped_mean_mpjpe:
            logger.info("  dropped by mean_mpjpe > %.4f:  %d frames",
                         mean_mpjpe_threshold, dropped_mean_mpjpe)
        if dropped_missing:
            logger.info("  dropped (missing metadata):    %d frames", dropped_missing)

        return {
            "obs": torch.from_numpy(obs).float(),
            "actions": torch.from_numpy(actions).float(),
            "action_mask": torch.from_numpy(action_mask).float(),
        }

    def sample_train_bc_batch(
        self, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch from the train BC dataset.

        Returns (obs, actions) tensors on *device*.
        """
        ds = self._bc_train
        n = ds["obs"].shape[0]
        idx = torch.randint(n, (self._bc_batch_size,))
        return ds["obs"][idx].to(device), ds["actions"][idx].to(device)

    def iter_test_bc_batches(
        self, batch_size: int, device: torch.device,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Yield (obs, actions) batches over the full test BC dataset."""
        ds = self._bc_test
        n = ds["obs"].shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield (
                ds["obs"][start:end].to(device),
                ds["actions"][start:end].to(device),
            )

    def compute_bc_loss(
        self, pred: torch.Tensor, target: torch.Tensor, device: torch.device,
    ) -> torch.Tensor:
        """Masked MSE: only legs/back actuators, arms excluded."""
        mask = self._bc_train["action_mask"].to(device)
        diff = (pred - target) ** 2
        return (diff * mask).sum(dim=1).mean() / mask.sum()
