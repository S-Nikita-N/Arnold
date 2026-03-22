"""
Action Parser для Arnold.

Зеркало ObservationParser для action-side:
- action_signatures: List[Tuple[str, ...]] для каждой мышцы
- грануляция мышц (анатомическая агрегация фасций)
- MuscleGrouping для ActionTokenizer
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import logging

logger = logging.getLogger(__name__)


# ============================================================
#  MuscleGrouping
# ============================================================

@dataclass
class MuscleGrouping:
    """
    Результат грануляции мышц.

    - muscle_to_group: muscle_name -> group_id (str)
    - groups: group_id -> list of muscle names (в порядке индексов)
    - group_order: порядок group_id для консистентной индексации
    """
    muscle_to_group: Dict[str, str] = field(default_factory=dict)
    groups: Dict[str, List[str]] = field(default_factory=dict)
    group_order: List[str] = field(default_factory=list)

    @property
    def n_groups(self) -> int:
        return len(self.group_order)

    @property
    def n_muscles(self) -> int:
        return len(self.muscle_to_group)

    def get_group_index(self, group_id: str) -> int:
        try:
            return self.group_order.index(group_id)
        except ValueError:
            return -1

    def get_muscle_indices_in_group(self, group_id: str) -> List[int]:
        """Индексы мышц внутри группы (относительные)."""
        return list(range(len(self.groups.get(group_id, []))))


# ============================================================
#  Анатомическая грануляция — таблицы маппинга
# ============================================================

# Торс: regex-паттерны для фасций → анатомическая группа
_ANATOMICAL_PREFIXES: List[Tuple[str, str]] = [
    # Pectoralis
    (r"^PECM[123]_", "pectoralis"),
    # Deltoids
    (r"^DELT[123]_", "deltoids"),
    # Latissimus
    (r"^LAT[123]_", "latissimus"),
    # Gluteus
    (r"^glmax[123]_", "gluteus_max"),
    (r"^glmed[123]_", "gluteus_med"),
    (r"^glmin[123]_", "gluteus_min"),
    # Iliocostalis (IL_*)
    (r"^IL_L[1-4]_", "iliocostalis_lumbar"),
    (r"^IL_R[0-9]+_", "iliocostalis_thoracic"),
    # Longissimus (LTpT_*, LTpL_*)
    (r"^LTpT_", "longissimus"),
    (r"^LTpL_", "longissimus_lateral"),
    # Multifidus
    (r"^MF_", "multifidus"),
    # Quadratus lumborum
    (r"^QL_", "quadratus_lumborum"),
    # External/Internal oblique
    (r"^EO[1-6]_", "external_oblique"),
    (r"^IO[1-6]_", "internal_oblique"),
    # Rectus abdominis
    (r"^rect_abd_", "rectus_abdominis"),
    # Erector spinae (Ps_* - paraspinal)
    (r"^Ps_", "paraspinal"),
    # Torso simple (ercspn, intobl, extobl)
    (r"^ercspn_", "erector_spinae_cervical"),
    (r"^intobl_", "internal_oblique_simple"),
    (r"^extobl_", "external_oblique_simple"),
]

# Конечности: base_name → функциональная группа
# (фасции DELT/PECM/LAT/glmax/glmed/glmin обрабатываются regex-ами выше)
_ANATOMICAL_LEG_ARM: Dict[str, str] = {
    "addbrev": "hip_adductors",
    "addlong": "hip_adductors",
    "addmagDist": "hip_adductors",
    "addmagIsch": "hip_adductors",
    "addmagMid": "hip_adductors",
    "addmagProx": "hip_adductors",
    "bflh": "knee_flexors",
    "bfsh": "knee_flexors",
    "semimem": "knee_flexors",
    "semiten": "knee_flexors",
    "gaslat": "ankle_plantarflexors",
    "gasmed": "ankle_plantarflexors",
    "soleus": "ankle_plantarflexors",
    "tibpost": "ankle_plantarflexors",
    "perlong": "ankle_plantarflexors",
    "perbrev": "ankle_plantarflexors",
    "tibant": "ankle_dorsiflexors",
    "edl": "ankle_dorsiflexors",
    "ehl": "ankle_dorsiflexors",
    "fdl": "ankle_plantarflexors",
    "fhl": "ankle_plantarflexors",
    "psoas": "hip_flexors",
    "iliacus": "hip_flexors",
    "recfem": "hip_flexors",
    "sart": "hip_flexors",
    "vaslat": "knee_extensors",
    "vasmed": "knee_extensors",
    "vasint": "knee_extensors",
    "tfl": "hip_abductors",
    "piri": "hip_abductors",
    "grac": "hip_adductors",
    # Arms (fascicle muscles like DELT/PECM/LAT are handled by regex)
    "CORB": "shoulder_flexors",
    "TMAJ": "shoulder_extensors",
    "TRIlong": "elbow_extensors",
    "SUPSP": "shoulder_abductors",
    "INFSP": "rotator_cuff",
    "SUBSC": "rotator_cuff",
    "TMIN": "rotator_cuff",
    "BIClong": "elbow_flexors",
    "BICshort": "elbow_flexors",
    "BRA": "elbow_flexors",
    "BRD": "elbow_flexors",
    "TRIlat": "elbow_extensors",
    "TRImed": "elbow_extensors",
    "ANC": "elbow_extensors",
    "SUP": "pronators_supinators",
}


# ============================================================
#  ActionParser
# ============================================================

class ActionParser:
    """
    Парсит action space для Arnold.

    Зеркало ObservationParser: строит action_signatures для vocabulary lookup
    и предоставляет грануляцию мышц для ActionTokenizer.
    """

    def __init__(self, muscle_names: List[str]):
        """
        Args:
            muscle_names: Список имён мышц (actuator names из env).
                          Порядок должен совпадать с порядком в модели MuJoCo.
        """
        self.muscle_names = muscle_names
        self.num_muscles = len(muscle_names)
        self.action_signatures = self._build_action_signatures()

    @classmethod
    def from_env(cls, env) -> "ActionParser":
        """
        Создаёт парсер из среды MuJoCo.

        Args:
            env: Среда MyoLegsIm или аналог с mj_model
        """
        muscle_names = []
        for i in range(env.mj_model.nu):
            muscle_names.append(env.mj_model.actuator(i).name)
        return cls(muscle_names)

    @staticmethod
    def _parse_side(name: str) -> Tuple[str, str]:
        """Извлекает base_name и side (r/l/c)."""
        if name.endswith("_r"):
            return name[:-2], "r"
        if name.endswith("_l"):
            return name[:-2], "l"
        return name, "c"

    def _build_action_signatures(self) -> List[Tuple[str, ...]]:
        """Строит signatures для action outputs (muscle activations)."""
        signatures = []
        for muscle in self.muscle_names:
            base, side = self._parse_side(muscle)
            signatures.append((base, side, "muscle", "activation"))
        return signatures

    def get_muscle_grouping(self, strategy: str = "anatomical") -> MuscleGrouping:
        """
        Грануляция мышц по выбранной стратегии.

        Args:
            strategy: "anatomical" — анатомическая агрегация фасций

        Returns:
            MuscleGrouping
        """
        if strategy == "anatomical":
            return self._granulate_anatomical()
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Supported: 'anatomical'."
        )

    def _anatomical_group(self, muscle_name: str) -> str:
        """Определяет анатомическую группу для одной мышцы."""
        base, side = self._parse_side(muscle_name)
        side_suffix = f"_{side}" if side != "c" else ""

        for pattern, group_base in _ANATOMICAL_PREFIXES:
            if re.match(pattern, muscle_name):
                return group_base + side_suffix

        for prefix, group_base in _ANATOMICAL_LEG_ARM.items():
            if base.startswith(prefix) or base == prefix:
                return group_base + side_suffix

        return f"other{side_suffix}"

    def _granulate_anatomical(self) -> MuscleGrouping:
        """
        Анатомическая агрегация фасций.
        Фасции одной мышцы объединяются в одну группу.
        """
        muscle_to_group: Dict[str, str] = {}
        groups: Dict[str, List[str]] = {}

        for m in self.muscle_names:
            g = self._anatomical_group(m)
            muscle_to_group[m] = g
            groups.setdefault(g, []).append(m)

        group_order = sorted(groups.keys())
        logger.info(f"Anatomical: {len(self.muscle_names)} muscles -> {len(group_order)} groups")
        return MuscleGrouping(
            muscle_to_group=muscle_to_group,
            groups=groups,
            group_order=group_order,
        )
