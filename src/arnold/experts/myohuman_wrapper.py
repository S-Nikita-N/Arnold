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
    ):
        """
        Args:
            cfg_path: Путь к директории cfg MyoHuman (None для default)
            expert_cfg: Готовый DictConfig (для multiprocessing, вместо cfg_path)
            checkpoint_epoch: Эпоха чекпоинта эксперта (0 = без эксперта, -1 = latest)
            device: Устройство ("cpu" или "cuda")
            overrides: Hydra overrides
            mode: "train" или "valid" — определяет motion file и настройки
        """
        self.mode = mode
        self.device = torch.device(device)
        self._expert_policy = None  # Загружается опционально

        # Сохраняем cwd и переключаемся на MyoHuman root
        # (MyoHuman использует относительные пути от своего корня)
        original_cwd = os.getcwd()
        os.chdir(MYOHUMAN_ROOT)

        try:
            # Импортируем после добавления путей
            from myohuman.env.myolegs_im import MyoLegsIm

            # Загружаем или используем готовый конфиг
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
                "or set imitation_weight=0 for PPO-only training."
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
