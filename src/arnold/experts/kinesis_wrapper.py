"""
Обёртка для эксперта Kinesis (MyoLegs).
Требует установленный подмодуль `arnold/experts/Kinesis` и рабочие модели.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

from typing import Tuple, Iterator
from omegaconf import DictConfig
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


# Добавляем путь к Kinesis src для корректных импортов
KINESIS_ROOT = Path(__file__).parent / "Kinesis"
KINESIS_SRC = KINESIS_ROOT / "src"
KINESIS_CFG = KINESIS_ROOT / "cfg"
KINESIS_DATA = KINESIS_ROOT / "data"

# Вставляем в начало sys.path
if str(KINESIS_ROOT) not in sys.path:
    sys.path.insert(0, str(KINESIS_ROOT))
if str(KINESIS_SRC) not in sys.path:
    sys.path.insert(0, str(KINESIS_SRC))


def load_kinesis_config(
    config_dir: str = None,
    overrides: list = None,
) -> DictConfig:
    """
    Загружает полный конфиг Kinesis через Hydra.
    
    Args:
        config_dir: Путь к директории cfg Kinesis (по умолчанию - из submodule)
        overrides: Список Hydra overrides
    
    Returns:
        DictConfig с полным конфигом
    """
    if config_dir is None:
        config_dir = str(KINESIS_CFG)
    
    if overrides is None:
        overrides = []
    
    # Очищаем предыдущую инициализацию Hydra если есть
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    # Инициализируем Hydra с абсолютным путём
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    
    return cfg


class KinesisWrapper:
    """
    Простой интерфейс к эксперту Kinesis (MyoLegs imitation).
    - Загружает агент из подмодуля Kinesis с указанным конфигом/чекпоинтом.
    - Предоставляет get_expert_action(flat_obs) → np.ndarray (в порядке ctrl env).
    """

    def __init__(
        self,
        cfg_path: str = None,
        expert_cfg: DictConfig = None,
        checkpoint_epoch: int = -1,
        device: str = "cpu",
        overrides: list = None,
        mode: str = "train",
    ):
        """
        Args:
            cfg_path: Путь к директории cfg Kinesis (None для default)
            expert_cfg: Готовый DictConfig (для multiprocessing, вместо cfg_path)
            checkpoint_epoch: Эпоха чекпоинта (-1 для latest)
            device: Устройство ("cpu" или "cuda")
            overrides: Hydra overrides (e.g., ["run=eval_run"])
            mode: "train" или "valid" — определяет какой motion_file использовать
                  train: kit_train_motion_dict.pkl
                  valid: kit_test_motion_dict.pkl
        """
        self.mode = mode
        
        # Сохраняем текущую директорию и меняем на Kinesis root
        # (Kinesis использует относительные пути от своего корня)
        original_cwd = os.getcwd()
        os.chdir(KINESIS_ROOT)
        
        try:
            # Импортируем после добавления путей
            from agents.agent_im import AgentIM
            from learning.learning_utils import to_test
            
            # Используем готовый cfg или загружаем через Hydra
            if expert_cfg is not None:
                # Используем готовый конфиг (для multiprocessing workers)
                self.cfg = expert_cfg
            else:
                # Загружаем конфиг через Hydra
                # Выбираем run config в зависимости от mode
                if mode == "valid":
                    run_config = "run=eval_run"  # kit_test_motion_dict.pkl
                else:
                    run_config = "run=train_run"  # kit_train_motion_dict.pkl
                
                default_overrides = [
                    run_config,
                    "headless=True",
                    "no_log=True",
                    "epoch=-1",  # Важно! Пропускаем загрузку expert_path в PolicyMOE.__init__
                    "run.test=True",  # Но ставим test=True для eval mode
                    "run.im_eval=False",  # Это триггерит sample_motions в load_checkpoint
                ]
                if overrides:
                    default_overrides.extend(overrides)
                
                cfg_dir = cfg_path if cfg_path else str(KINESIS_CFG)
                self.cfg = load_kinesis_config(config_dir=cfg_dir, overrides=default_overrides)
            
            self.device = torch.device(device)
            
            # Устанавливаем output_dir для загрузки модели
            models_dir = KINESIS_DATA / "trained_models" / "kinesis-moe-imitation"
            self.cfg.output_dir = str(models_dir)
            
            # Инициализируем агента (он сам поднимет env внутри)
            self.agent = AgentIM(
                cfg=self.cfg,
                dtype=torch.float32,
                device=self.device,
                training=False,
                checkpoint_epoch=checkpoint_epoch,
            )
            to_test(self.agent.policy_net)

            # env доступен через свойство self.env -> self.agent.env
            self.action_dim = self.agent.env.action_space.shape[0]
            self.obs_dim = self.agent.env.observation_space.shape[0]
            # Границы action_space для preprocess (clip+rescale) как у эксперта
            self.actions_low = self.agent.env.action_space.low.copy()
            self.actions_high = self.agent.env.action_space.high.copy()
        
        finally:
            # Восстанавливаем директорию
            os.chdir(original_cwd)

    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        Сброс среды эксперта и возврат obs.
        """
        obs, info = self.agent.env.reset()
        return obs, info
    
    def forward_motions(self) -> Iterator[int]:
        """
        Итератор по всем движениям в библиотеке.
        Каждая итерация загружает следующее движение.
        После yield нужно вызвать reset() для инициализации эпизода.
        
        Yields:
            int: Индекс текущего движения.
        """
        return self.agent.env.forward_motions()

    def get_expert_action(self, flat_obs: np.ndarray) -> np.ndarray:
        """
        Получить действие эксперта по плоскому obs (в порядке env observation_space).
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(flat_obs).to(self.device).float()
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            action = self.agent.policy_net.select_action(obs_t, mean_action=True)[0]
            return action.cpu().numpy().squeeze()

    def preprocess_actions(self, action: np.ndarray) -> np.ndarray:
        """
        Clip и rescale действий как в Kinesis Agent.preprocess_actions (clip_actions=True).
        Среда ожидает действия в [actions_low, actions_high]; политика студента (Arnold)
        выдаёт неограниченный Gaussian — без clip в env уходят невалидные значения.
        """
        action = np.clip(action.astype(np.float32), self.actions_low, self.actions_high)
        # rescale_actions(low, high, x): x * (high-low)/2 + (high+low)/2 — для [-1,1] это id
        d = (self.actions_high - self.actions_low) / 2.0
        m = (self.actions_low + self.actions_high) / 2.0
        return action * d + m

    def step(self, action: np.ndarray):
        """Проброс шага в среду эксперта. Действия предварительно clip+rescale как у эксперта."""
        action = self.preprocess_actions(action)
        next_obs, reward, terminated, truncated, info = self.agent.env.step(action)
        info['r_body_pos'] = info['r_body_pos'][0]
        info['r_vel'] = info['r_vel'][0]
        return next_obs, reward, terminated, truncated, info
    
    @property
    def env(self):
        """Доступ к среде эксперта."""
        return self.agent.env
    
    @property
    def num_motions(self) -> int:
        """Количество загруженных движений."""
        return self.agent.env.motion_lib.num_all_motions()
    
    def sample_motions(self, num_motions: int = None) -> None:
        """
        Пересэмплирует движения из библиотеки.
        
        Args:
            num_motions: Количество движений для загрузки (None = все)
        """
        if hasattr(self.agent.env, 'sample_motions'):
            self.agent.env.sample_motions()
