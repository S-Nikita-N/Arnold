"""
Грануляция мышц для action decoder.

Стратегия 1: Анатомическая агрегация фасций (fascia → muscle group)
Стратегия 2: Функциональные синергии (по действию на сустав)

Обе стратегии возвращают MuscleGrouping с group_id для каждой мышцы.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)


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


# --- Стратегия 1: Анатомическая агрегация ---
# Паттерны: префикс мышцы -> group_id
# Основано на muscle_grouping.md

ANATOMICAL_PREFIXES: List[Tuple[str, str]] = [
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

# Мышцы ног и рук - по базовому имени (без _r/_l)
ANATOMICAL_LEG_ARM: Dict[str, str] = {
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
    "fdl": "ankle_dorsiflexors",
    "fhl": "ankle_dorsiflexors",
    "psoas": "hip_flexors",
    "iliacus": "hip_flexors",
    "recfem": "hip_flexors",
    "sart": "hip_flexors",
    "vaslat": "knee_extensors",
    "vasmed": "knee_extensors",
    "vasint": "knee_extensors",
    "glmax1": "hip_extensors",
    "glmax2": "hip_extensors",
    "glmax3": "hip_extensors",
    "glmed1": "hip_abductors",
    "glmed2": "hip_abductors",
    "glmed3": "hip_abductors",
    "glmin1": "hip_abductors",
    "glmin2": "hip_abductors",
    "glmin3": "hip_abductors",
    "tfl": "hip_abductors",
    "piri": "hip_abductors",
    "grac": "hip_adductors",
    # Arms
    "DELT1": "shoulder_flexors",
    "DELT2": "shoulder_abductors",
    "DELT3": "shoulder_extensors",
    "PECM1": "shoulder_flexors",
    "PECM2": "shoulder_flexors",
    "PECM3": "shoulder_flexors",
    "CORB": "shoulder_flexors",
    "LAT1": "shoulder_extensors",
    "LAT2": "shoulder_extensors",
    "LAT3": "shoulder_extensors",
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


def _parse_side(name: str) -> Tuple[str, str]:
    """Извлекает base_name и side (r/l/c)."""
    if name.endswith("_r"):
        return name[:-2], "r"
    if name.endswith("_l"):
        return name[:-2], "l"
    return name, "c"


def _anatomical_group(muscle_name: str) -> str:
    """Определяет анатомическую группу для одной мышцы."""
    base, side = _parse_side(muscle_name)
    side_suffix = f"_{side}" if side != "c" else ""

    for pattern, group_base in ANATOMICAL_PREFIXES:
        if re.match(pattern, muscle_name):
            return group_base + side_suffix

    for prefix, group_base in ANATOMICAL_LEG_ARM.items():
        if base.startswith(prefix) or base == prefix:
            return group_base + side_suffix

    return f"other{side_suffix}"


# --- Стратегия 2: Функциональные синергии ---
# Группы по действию на сустав (из muscle_grouping.md)

FUNCTIONAL_GROUPS: Dict[str, List[str]] = {
    "hip_flexors": ["psoas", "iliacus", "recfem", "sart"],
    "hip_extensors": ["glmax1", "glmax2", "glmax3", "bflh", "semimem", "semiten"],
    "hip_adductors": ["addbrev", "addlong", "addmagProx", "addmagMid", "addmagDist", "addmagIsch", "grac"],
    "hip_abductors": ["glmed1", "glmed2", "glmed3", "glmin1", "glmin2", "glmin3", "tfl", "piri"],
    "knee_extensors": ["recfem", "vaslat", "vasmed", "vasint"],
    "knee_flexors": ["bflh", "bfsh", "semimem", "semiten", "gaslat", "gasmed"],
    "ankle_plantarflexors": ["gaslat", "gasmed", "soleus", "tibpost", "perlong", "perbrev"],
    "ankle_dorsiflexors": ["tibant", "edl", "ehl"],
    "spine_flexors": ["rect_abd", "EO", "IO"],
    "spine_extensors": ["IL", "LTpT", "LTpL", "MF", "QL", "Ps"],
    "shoulder_flexors": ["PECM1", "PECM2", "PECM3", "CORB", "DELT1"],
    "shoulder_extensors": ["LAT1", "LAT2", "LAT3", "TMAJ", "DELT3", "TRIlong"],
    "shoulder_abductors": ["DELT2", "SUPSP"],
    "rotator_cuff": ["INFSP", "SUBSC", "TMIN"],
    "elbow_flexors": ["BIClong", "BICshort", "BRA", "BRD"],
    "elbow_extensors": ["TRIlong", "TRIlat", "TRImed", "ANC"],
}


def _functional_group(muscle_name: str) -> str:
    """Определяет функциональную группу для одной мышцы."""
    base, side = _parse_side(muscle_name)
    side_suffix = f"_{side}" if side != "c" else ""

    # Spine extensors (все фасции спины)
    if (base.startswith("IL_") or base.startswith("LTpT") or base.startswith("LTpL") or
            base.startswith("MF_") or base.startswith("QL_") or base.startswith("Ps_") or
            base.startswith("ercspn") or base.startswith("intobl") or base.startswith("extobl")):
        return "spine_extensors" + side_suffix
    if base.startswith("EO") or base.startswith("IO") or base.startswith("rect_abd"):
        return "spine_flexors" + side_suffix

    # Legs & arms - точное совпадение по префиксу
    for group_id, muscle_bases in FUNCTIONAL_GROUPS.items():
        if group_id in ("spine_flexors", "spine_extensors"):
            continue
        for mb in muscle_bases:
            if base == mb or (len(mb) > 2 and base.startswith(mb)):
                return group_id + side_suffix

    return f"other{side_suffix}"


def granulate_anatomical(muscle_names: List[str]) -> MuscleGrouping:
    """
    Стратегия 1: Анатомическая агрегация фасций.
    Фасции одной мышцы объединяются в одну группу.
    """
    muscle_to_group: Dict[str, str] = {}
    groups: Dict[str, List[str]] = {}

    for m in muscle_names:
        g = _anatomical_group(m)
        muscle_to_group[m] = g
        groups.setdefault(g, []).append(m)

    group_order = sorted(groups.keys())
    logger.info(f"Anatomical: {len(muscle_names)} muscles -> {len(group_order)} groups")
    return MuscleGrouping(muscle_to_group=muscle_to_group, groups=groups, group_order=group_order)


def granulate_functional(muscle_names: List[str]) -> MuscleGrouping:
    """
    Стратегия 2: Функциональные синергии.
    Мышцы группируются по действию на сустав.
    """
    muscle_to_group: Dict[str, str] = {}
    groups: Dict[str, List[str]] = {}

    for m in muscle_names:
        g = _functional_group(m)
        muscle_to_group[m] = g
        groups.setdefault(g, []).append(m)

    group_order = sorted(groups.keys())
    logger.info(f"Functional: {len(muscle_names)} muscles -> {len(group_order)} groups")
    return MuscleGrouping(muscle_to_group=muscle_to_group, groups=groups, group_order=group_order)


def granulate(
    muscle_names: List[str],
    strategy: str = "anatomical",
) -> MuscleGrouping:
    """
    Грануляция мышц по выбранной стратегии.

    Args:
        muscle_names: Список имён мышц (порядок = порядок в модели)
        strategy: "anatomical" или "functional"

    Returns:
        MuscleGrouping
    """
    if strategy == "anatomical":
        return granulate_anatomical(muscle_names)
    if strategy == "functional":
        return granulate_functional(muscle_names)
    raise ValueError(f"Unknown strategy: {strategy}. Use 'anatomical' or 'functional'.")
