"""
Observation Parser для Arnold.

Конвертирует flat observation из среды Kinesis/MyoLegsIm в structured формат
для TransformerPolicy:
- obs_timeseries: [batch, n_obs_elements, history_len]
- obs_signatures: List[Tuple[str, ...]] для каждого элемента obs
- action_signatures: List[Tuple[str, ...]] для каждой мышцы
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ObservationSpec:
    """Спецификация одного компонента наблюдения."""
    name: str
    size: int
    signatures: List[Tuple[str, ...]] = field(default_factory=list)


class ObservationParser:
    """
    Парсит flat observation из среды Kinesis/MyoLegsIm в structured формат для Arnold.
    
    Отслеживает историю наблюдений и генерирует signatures для vocabulary lookup.
    
    Usage:
        parser = ObservationParser.from_env(env, history_len=5)
        
        obs, _ = env.reset()
        parser.reset(obs)
        
        for step in range(max_steps):
            obs_ts, obs_sigs = parser.get_observation()  # для Arnold
            action_sigs = parser.get_action_signatures()  # для Arnold
            
            # ... Arnold forward ...
            
            next_obs, reward, done, _, info = env.step(action)
            parser.update(next_obs)
    """
    
    def __init__(
        self,
        body_names: List[str],
        muscle_names: List[str],
        proprioceptive_inputs: List[str],
        task_inputs: List[str],
        track_bodies: List[str],
        num_bodies: int,
        num_muscles: int,
        history_len: int = 5,
    ):
        """
        Args:
            body_names: Список имён тел из env.body_names
            muscle_names: Список имён мышц (actuator names)
            proprioceptive_inputs: Из cfg.run.proprioceptive_inputs
            task_inputs: Из cfg.run.task_inputs
            track_bodies: Список tracked bodies для imitation
            num_bodies: Число тел
            num_muscles: Число мышц (actuators)
            history_len: Длина истории
        """
        self.body_names = body_names
        self.muscle_names = muscle_names
        self.proprioceptive_inputs = proprioceptive_inputs
        self.task_inputs = task_inputs
        self.track_bodies = track_bodies
        self.num_bodies = num_bodies
        self.num_muscles = num_muscles
        self.history_len = history_len
        
        # Строим структуру observations
        self.obs_specs: List[ObservationSpec] = []
        self._build_observation_specs()
        
        # Общее число элементов в flat observation и структурированных элементов
        self.flat_obs_size = sum(spec.size for spec in self.obs_specs)
        self.n_obs_elements = sum(len(spec.signatures) for spec in self.obs_specs)
        
        # Сигнатуры для actions (muscle activations)
        self.action_signatures = self._build_action_signatures()
        
        # History buffer
        self._history: Optional[np.ndarray] = None  # [n_obs_elements, history_len]
        self._obs_signatures: List[Tuple[str, ...]] = []
        self._flat_to_struct_map: List[Tuple[int, int]] = []  # (spec_idx, local_idx)
        
        # Строим flat signatures и mapping
        self._build_flat_signatures()
    
    @classmethod
    def from_env(cls, env, history_len: int = 5) -> "ObservationParser":
        """
        Создаёт парсер из среды Kinesis/MyoLegsIm.
        
        Args:
            env: Среда MyoLegsIm или аналог
            history_len: Длина истории
        """
        cfg = env.cfg
        
        # Body names
        body_names = env.body_names
        
        # Muscle names
        muscle_names = []
        for i in range(env.mj_model.nu):
            muscle_names.append(env.mj_model.actuator(i).name)
        
        # Track bodies (для imitation task)
        track_bodies = getattr(env, 'track_bodies', body_names[:7])
        
        return cls(
            body_names=body_names,
            muscle_names=muscle_names,
            proprioceptive_inputs=list(cfg.run.proprioceptive_inputs),
            task_inputs=list(cfg.run.task_inputs),
            track_bodies=track_bodies,
            num_bodies=len(body_names),
            num_muscles=env.mj_model.nu,
            history_len=history_len,
        )
    
    def _parse_side(self, name: str) -> Tuple[str, str]:
        """
        Извлекает базовое имя и сторону из имени.
        
        Returns:
            (base_name, side) где side: "r", "l" или "c" (center)
        """
        if name.endswith("_r"):
            return name[:-2], "r"
        elif name.endswith("_l"):
            return name[:-2], "l"
        else:
            return name, "c"
    
    def _build_observation_specs(self) -> None:
        """
        Строит спецификации observation компонентов на основе конфига.
        """
        # ==== Proprioceptive inputs ====
        
        if "root_height" in self.proprioceptive_inputs:
            self.obs_specs.append(ObservationSpec(
                name="root_height",
                size=1,
                signatures=[("root", "c", "height")]
            ))
        
        if "root_tilt" in self.proprioceptive_inputs:
            self.obs_specs.append(ObservationSpec(
                name="root_tilt",
                size=4,
                signatures=[
                    ("root", "c", "tilt", "x"),      # cos(roll)
                    ("root", "c", "tilt", "y"),      # sin(roll)
                    ("root", "c", "tilt", "qx"),     # cos(pitch)
                    ("root", "c", "tilt", "qy"),     # sin(pitch)
                ]
            ))
        
        if "local_body_pos" in self.proprioceptive_inputs:
            # 3 * (num_bodies - 1), исключая root
            sigs = []
            for body in self.body_names[1:]:  # skip root
                base, side = self._parse_side(body)
                for coord in ["x", "y", "z"]:
                    sigs.append((base, side, "position", coord))
            
            self.obs_specs.append(ObservationSpec(
                name="local_body_pos",
                size=3 * (self.num_bodies - 1),
                signatures=sigs
            ))
        
        if "local_body_rot" in self.proprioceptive_inputs:
            # 6 * num_bodies (tan-norm representation)
            sigs = []
            for body in self.body_names:
                base, side = self._parse_side(body)
                for i in range(6):
                    if i < 3:
                        coord = ["x", "y", "z"][i]
                        sigs.append((base, side, "rotation", "tangent", coord))
                    else:
                        coord = ["x", "y", "z"][i - 3]
                        sigs.append((base, side, "rotation", "normal", coord))
            
            self.obs_specs.append(ObservationSpec(
                name="local_body_rot",
                size=6 * self.num_bodies,
                signatures=sigs
            ))
        
        if "local_body_vel" in self.proprioceptive_inputs:
            # 3 * num_bodies
            sigs = []
            for body in self.body_names:
                base, side = self._parse_side(body)
                for coord in ["x", "y", "z"]:
                    sigs.append((base, side, "linear", "velocity", coord))
            
            self.obs_specs.append(ObservationSpec(
                name="local_body_vel",
                size=3 * self.num_bodies,
                signatures=sigs
            ))
        
        if "local_body_ang_vel" in self.proprioceptive_inputs:
            # 3 * num_bodies
            sigs = []
            for body in self.body_names:
                base, side = self._parse_side(body)
                for coord in ["x", "y", "z"]:
                    sigs.append((base, side, "angular", "velocity", coord))
            
            self.obs_specs.append(ObservationSpec(
                name="local_body_ang_vel",
                size=3 * self.num_bodies,
                signatures=sigs
            ))
        
        if "muscle_len" in self.proprioceptive_inputs:
            sigs = []
            for muscle in self.muscle_names:
                base, side = self._parse_side(muscle)
                sigs.append((base, side, "muscle", "length"))
            
            self.obs_specs.append(ObservationSpec(
                name="muscle_len",
                size=self.num_muscles,
                signatures=sigs
            ))
        
        if "muscle_vel" in self.proprioceptive_inputs:
            sigs = []
            for muscle in self.muscle_names:
                base, side = self._parse_side(muscle)
                sigs.append((base, side, "muscle", "velocity"))
            
            self.obs_specs.append(ObservationSpec(
                name="muscle_vel",
                size=self.num_muscles,
                signatures=sigs
            ))
        
        if "muscle_force" in self.proprioceptive_inputs:
            sigs = []
            for muscle in self.muscle_names:
                base, side = self._parse_side(muscle)
                sigs.append((base, side, "muscle", "force"))
            
            self.obs_specs.append(ObservationSpec(
                name="muscle_force",
                size=self.num_muscles,
                signatures=sigs
            ))
        
        if "feet_contacts" in self.proprioceptive_inputs:
            self.obs_specs.append(ObservationSpec(
                name="feet_contacts",
                size=4,
                signatures=[
                    ("calcn", "r", "contacts"),   # right heel
                    ("toes", "r", "contacts"),    # right toes
                    ("calcn", "l", "contacts"),   # left heel
                    ("toes", "l", "contacts"),    # left toes
                ]
            ))
        
        # ==== Task inputs (imitation) ====
        
        if "diff_local_body_pos" in self.task_inputs:
            sigs = []
            for body in self.track_bodies:
                base, side = self._parse_side(body)
                for coord in ["x", "y", "z"]:
                    sigs.append((base, side, "error", "position", coord))
            
            self.obs_specs.append(ObservationSpec(
                name="diff_local_body_pos",
                size=3 * len(self.track_bodies),
                signatures=sigs
            ))
        
        if "diff_local_vel" in self.task_inputs:
            sigs = []
            for body in self.track_bodies:
                base, side = self._parse_side(body)
                for coord in ["x", "y", "z"]:
                    sigs.append((base, side, "error", "velocity", coord))
            
            self.obs_specs.append(ObservationSpec(
                name="diff_local_vel",
                size=3 * len(self.track_bodies),
                signatures=sigs
            ))
        
        if "local_ref_body_pos" in self.task_inputs:
            sigs = []
            for body in self.track_bodies:
                base, side = self._parse_side(body)
                for coord in ["x", "y", "z"]:
                    sigs.append((base, side, "target", "position", coord))
            
            self.obs_specs.append(ObservationSpec(
                name="local_ref_body_pos",
                size=3 * len(self.track_bodies),
                signatures=sigs
            ))
        
        if "diff_muscle_len" in self.task_inputs:
            sigs = []
            for muscle in self.muscle_names:
                base, side = self._parse_side(muscle)
                sigs.append((base, side, "error", "length"))
            
            self.obs_specs.append(ObservationSpec(
                name="diff_muscle_len",
                size=self.num_muscles,
                signatures=sigs
            ))
        
        if "diff_muscle_vel" in self.task_inputs:
            sigs = []
            for muscle in self.muscle_names:
                base, side = self._parse_side(muscle)
                sigs.append((base, side, "error", "velocity"))
            
            self.obs_specs.append(ObservationSpec(
                name="diff_muscle_vel",
                size=self.num_muscles,
                signatures=sigs
            ))
    
    def _build_action_signatures(self) -> List[Tuple[str, ...]]:
        """Строит signatures для action outputs (muscle activations)."""
        sigs = []
        for muscle in self.muscle_names:
            base, side = self._parse_side(muscle)
            sigs.append((base, side, "muscle", "activation"))
        return sigs
    
    def _build_flat_signatures(self) -> None:
        """Строит flat список signatures и mapping."""
        self._obs_signatures = []
        self._flat_to_struct_map = []
        
        for spec_idx, spec in enumerate(self.obs_specs):
            for local_idx, sig in enumerate(spec.signatures):
                self._obs_signatures.append(sig)
                self._flat_to_struct_map.append((spec_idx, local_idx))
    
    def reset(self, initial_obs: np.ndarray) -> None:
        """
        Сбрасывает парсер с начальным наблюдением.
        
        Заполняет всю историю одним и тем же наблюдением.
        
        Args:
            initial_obs: Flat observation [flat_obs_size]
        """
        structured = self._flat_to_structured(initial_obs)
        # Заполняем историю одним obs
        self._history = np.tile(structured[:, np.newaxis], (1, self.history_len))
    
    def update(self, obs: np.ndarray) -> None:
        """
        Обновляет историю новым наблюдением.
        
        Args:
            obs: Flat observation [flat_obs_size]
        """
        if self._history is None:
            raise RuntimeError("Parser not initialized. Call reset() first.")
        
        structured = self._flat_to_structured(obs)
        # Сдвигаем историю и добавляем новое obs
        self._history = np.roll(self._history, -1, axis=1)
        self._history[:, -1] = structured
    
    def _flat_to_structured(self, flat_obs: np.ndarray) -> np.ndarray:
        """
        Конвертирует flat observation в structured (one timestep).
        
        Args:
            flat_obs: [flat_obs_size]
        
        Returns:
            [n_obs_elements] - каждый элемент это одно значение
        """
        structured = np.zeros(self.n_obs_elements, dtype=np.float32)
        
        flat_idx = 0
        struct_idx = 0
        for spec in self.obs_specs:
            for i in range(spec.size):
                structured[struct_idx] = flat_obs[flat_idx]
                flat_idx += 1
                struct_idx += 1
        
        return structured
    
    def get_observation(self, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, List[Tuple[str, ...]]]:
        """
        Возвращает observation для Arnold.
        
        Returns:
            obs_timeseries: [1, n_obs_elements, history_len] - tensor
            obs_signatures: List[Tuple[str, ...]] - signatures для vocabulary
        """
        if self._history is None:
            raise RuntimeError("Parser not initialized. Call reset() first.")
        
        # [n_obs_elements, history_len] -> [1, n_obs_elements, history_len]
        obs_ts = torch.from_numpy(self._history).float().unsqueeze(0).to(device)
        
        return obs_ts, self._obs_signatures
    
    def get_action_signatures(self) -> List[Tuple[str, ...]]:
        """Возвращает signatures для action outputs."""
        return self.action_signatures
    
    @property
    def obs_signatures(self) -> List[Tuple[str, ...]]:
        """Signatures для всех observation elements."""
        return self._obs_signatures


class ObservationParserSimple:
    """
    Упрощённый парсер, который не требует знания структуры observations.
    
    Использует generic signatures вида ("obs", idx) для каждого элемента.
    Подходит для быстрого прототипирования.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        history_len: int = 5,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.history_len = history_len
        
        # Generic signatures
        self._obs_signatures = [("position", "c", "value", str(i)) for i in range(obs_dim)]
        self._action_signatures = [("muscle", "c", "activation", str(i)) for i in range(action_dim)]
        
        self._history: Optional[np.ndarray] = None
    
    def reset(self, initial_obs: np.ndarray) -> None:
        self._history = np.tile(initial_obs[:, np.newaxis], (1, self.history_len))
    
    def update(self, obs: np.ndarray) -> None:
        if self._history is None:
            raise RuntimeError("Parser not initialized. Call reset() first.")
        self._history = np.roll(self._history, -1, axis=1)
        self._history[:, -1] = obs
    
    def get_observation(self, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, List[Tuple[str, ...]]]:
        if self._history is None:
            raise RuntimeError("Parser not initialized. Call reset() first.")
        obs_ts = torch.from_numpy(self._history).float().unsqueeze(0).to(device)
        return obs_ts, self._obs_signatures
    
    def get_action_signatures(self) -> List[Tuple[str, ...]]:
        return self._action_signatures
