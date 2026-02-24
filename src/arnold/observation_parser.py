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


@dataclass
class BodyGroup:
    """Группа наблюдений, объединённых по телу/мышце для body-level tokenization."""
    name: str
    side: str
    type: str
    indices: List[int]
    signature: Tuple[str, ...]


class ObservationParser:
    """
    Парсит flat observation из среды Kinesis/MyoLegsIm в structured формат для Arnold.
    Отслеживает историю наблюдений и генерирует signatures для vocabulary lookup.
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
        
        self.obs_specs = self._build_observation_specs()
        self.n_obs_elements = sum(spec.size for spec in self.obs_specs)
        self.action_signatures = self._build_action_signatures()
        self.obs_signatures = self._build_obs_signatures()
        self.history: Optional[np.ndarray] = None
    
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
    
    def _build_observation_specs(self) -> List[ObservationSpec]:
        """
        Строит спецификации observation компонентов на основе конфига.
        """
        specs = []
        # ==== Proprioceptive inputs ====
        
        if "root_height" in self.proprioceptive_inputs:
            specs.append(ObservationSpec(
                name="root_height",
                size=1,
                signatures=[("root", "c", "height")]
            ))
        
        if "root_tilt" in self.proprioceptive_inputs:
            specs.append(ObservationSpec(
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
            
            specs.append(ObservationSpec(
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
            
            specs.append(ObservationSpec(
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
            
            specs.append(ObservationSpec(
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
            
            specs.append(ObservationSpec(
                name="local_body_ang_vel",
                size=3 * self.num_bodies,
                signatures=sigs
            ))
        
        if "muscle_len" in self.proprioceptive_inputs:
            sigs = []
            for muscle in self.muscle_names:
                base, side = self._parse_side(muscle)
                sigs.append((base, side, "muscle", "length"))
            
            specs.append(ObservationSpec(
                name="muscle_len",
                size=self.num_muscles,
                signatures=sigs
            ))
        
        if "muscle_vel" in self.proprioceptive_inputs:
            sigs = []
            for muscle in self.muscle_names:
                base, side = self._parse_side(muscle)
                sigs.append((base, side, "muscle", "velocity"))
            
            specs.append(ObservationSpec(
                name="muscle_vel",
                size=self.num_muscles,
                signatures=sigs
            ))
        
        if "muscle_force" in self.proprioceptive_inputs:
            sigs = []
            for muscle in self.muscle_names:
                base, side = self._parse_side(muscle)
                sigs.append((base, side, "muscle", "force"))
            
            specs.append(ObservationSpec(
                name="muscle_force",
                size=self.num_muscles,
                signatures=sigs
            ))
        
        if "feet_contacts" in self.proprioceptive_inputs:
            specs.append(ObservationSpec(
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
            
            specs.append(ObservationSpec(
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
            
            specs.append(ObservationSpec(
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
            
            specs.append(ObservationSpec(
                name="local_ref_body_pos",
                size=3 * len(self.track_bodies),
                signatures=sigs
            ))
        
        if "diff_muscle_len" in self.task_inputs:
            sigs = []
            for muscle in self.muscle_names:
                base, side = self._parse_side(muscle)
                sigs.append((base, side, "error", "length"))
            
            specs.append(ObservationSpec(
                name="diff_muscle_len",
                size=self.num_muscles,
                signatures=sigs
            ))
        
        if "diff_muscle_vel" in self.task_inputs:
            sigs = []
            for muscle in self.muscle_names:
                base, side = self._parse_side(muscle)
                sigs.append((base, side, "error", "velocity"))
            
            specs.append(ObservationSpec(
                name="diff_muscle_vel",
                size=self.num_muscles,
                signatures=sigs
            ))

        return specs

    def _build_action_signatures(self) -> List[Tuple[str, ...]]:
        """Строит signatures для action outputs (muscle activations)."""
        signatures = []
        for muscle in self.muscle_names:
            base, side = self._parse_side(muscle)
            signatures.append((base, side, "muscle", "activation"))
        return signatures
    
    def _build_obs_signatures(self) -> List[Tuple[str, ...]]:
        """Строит flat список signatures."""
        signatures = []
        for spec in self.obs_specs:
            for sig in spec.signatures:
                signatures.append(sig)
        return signatures
    
    def reset(self, initial_obs: np.ndarray) -> None:
        """
        Сбрасывает парсер с начальным наблюдением.
        
        Заполняет всю историю одним и тем же наблюдением.
        
        Args:
            initial_obs: Flat observation [flat_obs_size]
        """
        obs_f32 = initial_obs.astype(np.float32)
        self.history = np.tile(obs_f32[:, np.newaxis], (1, self.history_len))
    
    def update(self, obs: np.ndarray) -> None:
        """
        Обновляет историю новым наблюдением.
        
        Args:
            obs: Flat observation
        """
        if self.history is None:
            raise RuntimeError("Parser not initialized. Call reset() first.")
        
        self.history[:, :-1] = self.history[:, 1:]
        self.history[:, -1] = obs
    
    def get_observation(
        self,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, List[Tuple[str, ...]]]:
        """
        Возвращает observation для Arnold.
        
        Returns:
            obs_timeseries: [1, n_obs_elements, history_len] - tensor
            obs_signatures: List[Tuple[str, ...]] - signatures для vocabulary
        """
        if self.history is None:
            raise RuntimeError("Parser not initialized. Call reset() first.")
        
        # [n_obs_elements, history_len] -> [1, n_obs_elements, history_len]
        obs_ts = torch.from_numpy(self.history).float().unsqueeze(0).to(device)
        
        return obs_ts, self.obs_signatures

    def get_body_groups(
        self,
        granularity: str = "per_body"
    ) -> List[BodyGroup]:
        """
        Строит группы наблюдений с настраиваемой гранулярностью.

        Granularity (от самого мелкого к самому крупному):
          "scalar"   — каждый скаляр = свой токен (~1583). Эквивалент SensoryEncoder.
          "per_spec" — группировка по (тело, тип_obs). position xyz = 1 токен.
          "per_body" — всё про одно тело/мышцу в один токен (~112).
        """
        if granularity == "scalar":
            return self._build_groups_scalar()
        elif granularity == "per_spec":
            return self._build_groups_per_spec()
        elif granularity == "per_body":
            return self._build_groups_per_body()
        else:
            raise ValueError(
                f"Unknown granularity '{granularity}'. "
                f"Choose from: 'scalar', 'per_spec', 'per_body'"
            )

    def _build_groups_scalar(self) -> List[BodyGroup]:
        """
        Каждый observation scalar — отдельный токен.
        Linear(history_len, embed_dim).
        """
        groups: List[BodyGroup] = []
        offset = 0
        for spec in self.obs_specs:
            for i in range(spec.size):
                if spec.signatures and i < len(spec.signatures):
                    sig = spec.signatures[i]
                else:
                    sig = (spec.name, str(i))
                groups.append(BodyGroup(
                    name=sig[0],
                    side=sig[1] if len(sig) > 1 else "c",
                    type="scalar",
                    indices=[offset + i],
                    signature=sig,
                ))
            offset += spec.size
        return groups

    def _build_groups_per_spec(self) -> List[BodyGroup]:
        """
        Группировка по (entity, obs_spec).
        Напр. femur_l position (3 скаляра) = 1 токен.
        Для feet_contacts - отдельный токен.
        """
        groups: List[BodyGroup] = []
        offset = 0
        for spec in self.obs_specs:
            if spec.signatures:
                entity_indices: dict = {}
                for i, sig in enumerate(spec.signatures):
                    key = (sig[0], sig[1] if len(sig) > 1 else "c")
                    entity_indices.setdefault(key, []).append(offset + i)
                for (name, side), indices in entity_indices.items():
                    groups.append(BodyGroup(
                        name=name, side=side,
                        type=spec.name,
                        indices=indices,
                        signature=(name, side, spec.name),
                    ))
            else:
                groups.append(BodyGroup(
                    name=spec.name, side="c",
                    type=spec.name,
                    indices=list(range(offset, offset + spec.size)),
                    signature=(spec.name, "c"),
                ))
            offset += spec.size
        return groups

    def _build_groups_per_body(self) -> List[BodyGroup]:
        """
        Все наблюдения одного тела/мышцы(кроме feet_contacts)
        объединены в один токен.
        """

        ROOT_SPECS = {
            "root_height",
            "root_tilt"
        }
        BODY_SPECS = {
            "local_body_pos",
            "local_body_rot",
            "local_body_vel",
            "local_body_ang_vel",
        }
        TASK_BODY_SPECS = {
            "diff_local_body_pos",
            "diff_local_vel",
            "local_ref_body_pos",
        }
        MUSCLE_SPECS = {
            "muscle_len",
            "muscle_vel",
            "muscle_force",
        }
        TASK_MUSCLE_SPECS = {
            "diff_muscle_len",
            "diff_muscle_vel",
        }

        root_indices: List[int] = []
        body_indices: dict = {}
        muscle_indices: dict = {}
        task_body_indices: dict = {}
        task_muscle_indices: dict = {}
        feet_indices: List[int] = []

        offset = 0
        for spec in self.obs_specs:
            if spec.name in ROOT_SPECS:
                for i in range(spec.size):
                    root_indices.append(offset + i)
            elif spec.name in BODY_SPECS:
                for i, sig in enumerate(spec.signatures):
                    name, side = sig[0], sig[1]
                    if name == "root":
                        root_indices.append(offset + i)
                    else:
                        body_indices.setdefault((name, side), []).append(offset + i)
            elif spec.name in MUSCLE_SPECS:
                for i, sig in enumerate(spec.signatures):
                    muscle_indices.setdefault((sig[0], sig[1]), []).append(offset + i)
            elif spec.name == "feet_contacts":
                for i in range(spec.size):
                    feet_indices.append(offset + i)
            elif spec.name in TASK_BODY_SPECS:
                for i, sig in enumerate(spec.signatures):
                    task_body_indices.setdefault((sig[0], sig[1]), []).append(offset + i)
            elif spec.name in TASK_MUSCLE_SPECS:
                for i, sig in enumerate(spec.signatures):
                    task_muscle_indices.setdefault((sig[0], sig[1]), []).append(offset + i)
            offset += spec.size

        groups: List[BodyGroup] = []
        if root_indices:
            groups.append(
                BodyGroup(
                    name="root",
                    side="c",
                    type="root",
                    indices=root_indices,
                    signature=("root", "c"),
                )
            )
        for (name, side), indices in body_indices.items():
            groups.append(BodyGroup(
                name=name,
                side=side,
                type="body",
                indices=indices,
                signature=(name, side),
            ))
        for (name, side), indices in muscle_indices.items():
            groups.append(BodyGroup(
                name=name,
                side=side,
                type="muscle",
                indices=indices,
                signature=(name, side, "muscle"),
            ))
        if feet_indices:
            groups.append(BodyGroup(
                name="contacts",
                side="c",
                type="feet",
                indices=feet_indices,
                signature=("contacts", "c"),
            ))
        for (name, side), indices in task_body_indices.items():
            groups.append(BodyGroup(
                name=name,
                side=side,
                type="task",
                indices=indices,
                signature=(name, side, "error"),
            ))
        for (name, side), indices in task_muscle_indices.items():
            groups.append(BodyGroup(
                name=name,
                side=side,
                type="task_muscle",
                indices=indices,
                signature=(name, side, "muscle", "error"),
            ))
        return groups
    
