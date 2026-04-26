"""
MyoKin Wrapper — MyoHuman env + Kinesis teacher.

Используется для обучения Arnold policy в среде MyoHuman (338 actuators)
с учителем Kinesis (290 actuators, legs/back). Руки (48 actuators)
перезаписываются фиксированным значением при step().

Два режима:
  1. OBC — с учителем (teacher_checkpoint задан, obc_imitation_weight > 0)
  2. PPO — без учителя (teacher_checkpoint=null, ppo_weight > 0)

В обоих случаях wrapper:
  - Создаёт MyoHuman env (338 actuators)
  - Перезаписывает arm actions в step()
  - Предоставляет action_mask для transfer reinit (290 legs/back = 1, 48 arms = 0)
"""

import os
import sys
import logging
from pathlib import Path

import mujoco
import numpy as np
import torch

from typing import Tuple, Iterator

# Paths to submodules
_EXPERTS_DIR = Path(__file__).resolve().parent
_MYOHUMAN_ROOT = _EXPERTS_DIR / "Myohuman"
_MYOHUMAN_SRC = _MYOHUMAN_ROOT / "src"
_MYOHUMAN_CFG = _MYOHUMAN_ROOT / "cfg"

if str(_MYOHUMAN_ROOT) not in sys.path:
    sys.path.insert(0, str(_MYOHUMAN_ROOT))
if str(_MYOHUMAN_SRC) not in sys.path:
    sys.path.insert(0, str(_MYOHUMAN_SRC))

from arnold.experts.myohuman_wrapper import load_myohuman_config  # noqa: E402

logger = logging.getLogger(__name__)

# Arm actuator indices in MyoHuman action space (338 total)
ARM_ACTION_START = 290
ARM_ACTION_END = 338


class MyoKinWrapper:
    """
    MyoHuman env (338 actuators) + optional Kinesis teacher (290 actuators).

    Интерфейс:
    - reset() → (obs, info)
    - step(action) → (obs, reward, terminated, truncated, info)
    - get_expert_action(flat_obs) → np.ndarray  (только если teacher загружен)
    - get_action_mask() → torch.Tensor  (338,) — 1.0 legs/back, 0.0 arms
    - forward_motions() → Iterator[int]
    - env property → MyoLegsIm
    """

    def __init__(
        self,
        cfg_path: str = None,
        checkpoint_epoch: int = 0,  # ignored, kept for interface compat
        device: str = "cpu",
        overrides: list = None,
        mode: str = "train",
        teacher_checkpoint: str = None,
        arm_action_override: float = 0.0,
    ):
        if overrides is None:
            overrides = []

        self.mode = mode
        self.device = torch.device(device)
        self.arm_action_override = arm_action_override
        self._teacher = None
        self._obs_adapter = None

        original_cwd = os.getcwd()
        os.chdir(_MYOHUMAN_ROOT)

        try:
            from myohuman.env.myolegs_im import MyoLegsIm

            if mode == "valid":
                run_config = "run=eval_run"
            else:
                run_config = "run=train_run"

            default_overrides = [
                run_config,
                "no_log=True",
                "test=True" if mode == "valid" else "test=False",
            ]
            default_overrides.extend(overrides)

            if not any(o.startswith("run.headless=") for o in overrides):
                default_overrides.append("run.headless=True")

            cfg_dir = cfg_path if cfg_path else str(_MYOHUMAN_CFG)
            self.cfg = load_myohuman_config(
                config_dir=cfg_dir, overrides=default_overrides,
            )
            self.cfg.project_root = str(_MYOHUMAN_ROOT)

            self._env = MyoLegsIm(self.cfg)

            self.action_dim = self._env.action_space.shape[0]
            self.obs_dim = self._env.observation_space.shape[0]
            self.actions_low = self._env.action_space.low.copy()
            self.actions_high = self._env.action_space.high.copy()

            # Load motions
            if not (mode == "valid" and self.cfg.run.im_eval):
                self._env.sample_motions()

            # Load Kinesis teacher if checkpoint provided
            if teacher_checkpoint:
                self._load_teacher(teacher_checkpoint)

        finally:
            os.chdir(original_cwd)

        # Action mask: 1.0 for legs/back (0:290), 0.0 for arms (290:338)
        self._action_mask = torch.zeros(self.action_dim, dtype=torch.float32)
        self._action_mask[:ARM_ACTION_START] = 1.0

        # Arm joint qpos indices in MyoHuman model (same as evaluate_kinesis_teacher)
        self._arm_qpos_slice = slice(27, 59)

    def _load_teacher(self, checkpoint_path: str) -> None:
        """Load frozen Kinesis MoE teacher."""
        from arnold.experts.scripts.evaluate_kinesis_teacher import (
            KinesisTeacher,
            KinesisObservationAdapter,
        )

        self._teacher = KinesisTeacher(checkpoint_path, device=self.device)
        mh_body_names = [
            self._env.mj_model.body(i).name
            for i in range(self._env.mj_model.nbody)
        ]
        self._obs_adapter = KinesisObservationAdapter(mh_body_names)
        logger.info(f"MyoKin teacher loaded from {checkpoint_path}")

    @property
    def has_expert(self) -> bool:
        return self._teacher is not None

    @property
    def env(self):
        return self._env

    @property
    def num_motions(self) -> int:
        return len(self._env._all_motion_ids)

    def get_action_mask(self) -> torch.Tensor:
        """(338,) float mask: 1.0 for legs/back, 0.0 for arms."""
        return self._action_mask

    def reset(self) -> Tuple[np.ndarray, dict]:
        obs, info = self._env.reset()
        # Zero out arm joints so arms start in neutral pose
        self._env.mj_data.qpos[self._arm_qpos_slice] = 0.0
        mujoco.mj_forward(self._env.mj_model, self._env.mj_data)
        obs = self._env.get_obs()
        return obs, info

    def step(self, action: np.ndarray):
        """Step with arm override: replace arm actions before env.step."""
        action = action.copy()
        action[ARM_ACTION_START:ARM_ACTION_END] = self.arm_action_override
        action = self.preprocess_actions(action)
        next_obs, reward, terminated, truncated, info = self._env.step(action)

        if "r_body_pos" in info and isinstance(info["r_body_pos"], np.ndarray):
            info["r_body_pos"] = info["r_body_pos"][0] if info["r_body_pos"].ndim > 0 else info["r_body_pos"]
        if "r_vel" in info and isinstance(info["r_vel"], np.ndarray):
            info["r_vel"] = info["r_vel"][0] if info["r_vel"].ndim > 0 else info["r_vel"]

        return next_obs, reward, terminated, truncated, info

    def preprocess_actions(self, action: np.ndarray) -> np.ndarray:
        """Clip and rescale actions (same as MyoHumanWrapper)."""
        action = np.clip(action.astype(np.float32), self.actions_low, self.actions_high)
        d = (self.actions_high - self.actions_low) / 2.0
        m = (self.actions_low + self.actions_high) / 2.0
        return action * d + m

    def get_expert_action(self, flat_obs: np.ndarray) -> np.ndarray:
        """Get 338-dim teacher action (arms=0) from Kinesis MoE."""
        if self._teacher is None:
            raise RuntimeError(
                "Kinesis teacher not loaded. Set teacher_checkpoint in config."
            )

        kin_obs = self._obs_adapter.build_obs(self._env)
        kin_obs_t = torch.from_numpy(kin_obs).unsqueeze(0).to(self.device)
        actions_mh, _ = self._teacher.forward_batch(kin_obs_t)
        return actions_mh.squeeze(0).cpu().numpy()

    def forward_motions(self) -> Iterator[int]:
        return self._env.forward_motions()

    def sample_motions(self, num_motions: int = None) -> None:
        if hasattr(self._env, "sample_motions"):
            self._env.sample_motions()
