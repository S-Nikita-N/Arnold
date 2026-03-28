"""
Action Parser для Arnold.

Стратегии грануляции:
  "anatomical"  — чистая Strategy 1: только фасции одной мышцы сливаются,
                   индивидуальные мышцы конечностей остаются синглтонами.
  "functional"  — Strategy 2: группировка по функции сустава
                   (spine_flexors, hip_extensors, shoulder_flexors, …)
  "hybrid"      — анатомические фасции для торса + функциональные группы для конечностей.
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

    - muscle_to_group: muscle_name -> group_id (str), e.g. "pectoralis_r"
    - groups: group_id -> list of muscle names (в порядке индексов)
    - group_order: порядок group_id для консистентной индексации
    - granule_base_map: group_id -> granule base name (без side suffix),
                        e.g. "pectoralis_r" -> "pectoralis"
    """
    groups: Dict[str, List[str]] = field(default_factory=dict)
    group_order: List[str] = field(default_factory=list)
    muscle_to_group: Dict[str, str] = field(default_factory=dict)
    granule_base_map: Dict[str, str] = field(default_factory=dict)

    @property
    def n_groups(self) -> int:
        return len(self.group_order)

    @property
    def n_muscles(self) -> int:
        return len(self.muscle_to_group)

    @staticmethod
    def granule_base_from_group_id(group_id: str) -> str:
        """Извлекает granule base (без _r/_l суффикса) из group_id."""
        if group_id.endswith("_r") or group_id.endswith("_l"):
            return group_id[:-2]
        return group_id

    @staticmethod
    def parse_side_from_group_id(group_id: str) -> str:
        """Извлекает side (r/l/c) из group_id."""
        if group_id.endswith("_r"):
            return "r"
        if group_id.endswith("_l"):
            return "l"
        return "c"


# ============================================================
#  Таблицы маппинга: regex фасций + словари конечностей
# ============================================================

# --- Anatomical/Hybrid: regex фасций торса ---
ANATOMICAL_TORSO_REGEX_MAP: List[Tuple[str, str]] = [
    (r"^PECM[123]_", "pectoralis"),  # фасции большой грудной — большая грудная мышца
    (r"^DELT[123]_", "deltoids"),  # фасции дельтовидной — дельтовидная мышца
    (r"^LAT[123]_", "latissimus"),  # фасции широчайшей — широчайшая мышца спины
    (r"^glmax[123]_", "gluteus_max"),  # фасции большой ягодичной — большая ягодичная мышца
    (r"^glmed[123]_", "gluteus_med"),  # фасции средней ягодичной — средняя ягодичная мышца
    (r"^glmin[123]_", "gluteus_min"),  # фасции малой ягодичной — малая ягодичная мышца
    (r"^IL_", "iliocostalis"),  # фасции IL_* — подвздошно-рёберная мышца
    (r"^LTpT_", "longissimus"),  # фасции LTpT_* (грудной ход) — длиннейшая мышца спины
    (r"^LTpL_", "longissimus"),  # фасции LTpL_* (поясничный ход) — длиннейшая мышца спины
    (r"^MF_", "multifidus"),  # фасции MF_* — многораздельная мышца
    (r"^QL_", "quadratus_lumborum"),  # фасции QL_* — квадратная мышца поясницы
    (r"^EO[1-6]_", "external_oblique"),  # фасции EO_* — наружная косая мышца живота
    (r"^IO[1-6]_", "internal_oblique"),  # фасции IO_* — внутренняя косая мышца живота
    (r"^rect_abd_", "rectus_abdominis"),  # rect_abd — прямая мышца живота
    (r"^Ps_", "paraspinal"),  # фасции Ps_* — параспинальные сегментарные (не одна мышца → одна гранула)
    (r"^ercspn_", "erector_spinae_cervical"),  # ercspn — шейный разгибатель, упрощённая мышца в модели
    (r"^intobl_", "internal_oblique_simple"),  # intobl — внутренняя косая (упрощ.), одна мышца в модели
    (r"^extobl_", "external_oblique_simple"),  # extobl — наружная косая (упрощ.), одна мышца в модели
]

# --- Functional: regex для торса по функции ---
FUNCTIONAL_TORSO_REGEX_MAP: List[Tuple[str, str]] = [
    (r"^EO[1-6]_", "spine_flexors"),  # наружная косая живота, фасции — сгибатели позвоночника (корпус)
    (r"^IO[1-6]_", "spine_flexors"),  # внутренняя косая живота, фасции — сгибатели позвоночника (корпус)
    (r"^rect_abd_", "spine_flexors"),  # прямая мышца живота — сгибатели позвоночника (корпус)
    (r"^intobl_", "spine_flexors"),  # внутр. косая упрощ. — сгибатели позвоночника (корпус)
    (r"^extobl_", "spine_flexors"),  # наруж. косая упрощ. — сгибатели позвоночника (корпус)
    (r"^IL_", "spine_extensors"),  # подвздошно-рёберная, фасции — разгибатели позвоночника (корпус)
    (r"^LTpT_", "spine_extensors"),  # длиннейшая грудной ход — разгибатели позвоночника (корпус)
    (r"^LTpL_", "spine_extensors"),  # длиннейшая поясничный ход — разгибатели позвоночника (корпус)
    (r"^MF_", "spine_extensors"),  # многораздельная — разгибатели позвоночника (корпус)
    (r"^QL_", "spine_extensors"),  # квадратная поясничная — разгибатели позвоночника (корпус)
    (r"^Ps_", "spine_extensors"),  # параспинальные — разгибатели позвоночника (корпус)
    (r"^ercspn_", "spine_extensors"),  # шейный столб разгибателей — разгибатели позвоночника (корпус)
    (r"^PECM[123]_", "shoulder_flexors"),  # большая грудная, фасции — сгибатели плечевого сустава
    (r"^DELT1_", "shoulder_flexors"),  # дельтовидная передняя фасция — сгибатели плечевого сустава
    (r"^LAT[123]_", "shoulder_extensors"),  # широчайшая, фасции — разгибатели плечевого сустава
    (r"^DELT3_", "shoulder_extensors"),  # дельтовидная задняя фасция — разгибатели плечевого сустава
    (r"^DELT2_", "shoulder_abductors"),  # дельтовидная средняя фасция — отводящие мышцы плеча
    (r"^glmax[123]_", "hip_extensors"),  # большая ягодичная, фасции — разгибатели бедра в тазобедренном суставе
    (r"^glmed[123]_", "hip_abductors"),  # средняя ягодичная, фасции — отводящие мышцы бедра
    (r"^glmin[123]_", "hip_abductors"),  # малая ягодичная, фасции — отводящие мышцы бедра
]

# --- Hybrid: функциональные группы конечностей (торс — ANATOMICAL_TORSO_REGEX_MAP) ---
HYBRID_LIMB_MAP: Dict[str, str] = {
    "addbrev": "hip_adductors",  # короткая приводящая бедра — приводящие мышцы бедра
    "addlong": "hip_adductors",  # длинная приводящая бедра — приводящие мышцы бедра
    "addmagDist": "hip_adductors",  # приводящая большая дистальная часть — приводящие мышцы бедра
    "addmagIsch": "hip_adductors",  # приводящая большая седалищная часть — приводящие мышцы бедра
    "addmagMid": "hip_adductors",  # приводящая большая средняя часть — приводящие мышцы бедра
    "addmagProx": "hip_adductors",  # приводящая большая проксимальная часть — приводящие мышцы бедра
    "bflh": "knee_flexors",  # двуглавая бедра длинная головка — сгибатели коленного сустава
    "bfsh": "knee_flexors",  # двуглавая бедра короткая головка — сгибатели коленного сустава
    "semimem": "knee_flexors",  # полуперепончатая — сгибатели коленного сустава
    "semiten": "knee_flexors",  # полусухожильная — сгибатели коленного сустава
    "gaslat": "ankle_plantarflexors",  # икроножная латеральная — подошвенные сгибатели голеностопа
    "gasmed": "ankle_plantarflexors",  # икроножная медиальная — подошвенные сгибатели голеностопа
    "soleus": "ankle_plantarflexors",  # камбаловидная — подошвенные сгибатели голеностопа
    "tibpost": "ankle_plantarflexors",  # задняя большеберцовая — подошвенные сгибатели голеностопа
    "perlong": "ankle_plantarflexors",  # длинная малоберцовая — подошвенные сгибатели голеностопа
    "perbrev": "ankle_plantarflexors",  # короткая малоберцовая — подошвенные сгибатели голеностопа
    "tibant": "ankle_dorsiflexors",  # передняя большеберцовая — тыльные разгибатели голеностопа
    "edl": "ankle_dorsiflexors",  # длинный разгибатель пальцев стопы — тыльные разгибатели голеностопа
    "ehl": "ankle_dorsiflexors",  # длинный разгибатель большого пальца стопы — тыльные разгибатели голеностопа
    "fdl": "ankle_plantarflexors",  # длинный сгибатель пальцев стопы — подошвенные сгибатели голеностопа
    "fhl": "ankle_plantarflexors",  # длинный сгибатель большого пальца стопы — подошвенные сгибатели голеностопа
    "psoas": "hip_flexors",  # поясничный мышечный столб — сгибатели бедра в тазобедренном суставе
    "iliacus": "hip_flexors",  # подвздошная — сгибатели бедра в тазобедренном суставе
    "recfem": "hip_flexors",  # прямая мышца бедра — сгибатели бедра в тазобедренном суставе
    "sart": "hip_flexors",  # портняжная — сгибатели бедра в тазобедренном суставе
    "vaslat": "knee_extensors",  # латеральная широкая мышца бедра — разгибатели коленного сустава
    "vasmed": "knee_extensors",  # медиальная широкая мышца бедра — разгибатели коленного сустава
    "vasint": "knee_extensors",  # промежуточная широкая мышца бедра — разгибатели коленного сустава
    "tfl": "hip_abductors",  # широкая фасция бедра / тракт — отводящие мышцы бедра
    "piri": "hip_abductors",  # грушевидная — отводящие мышцы бедра
    "grac": "hip_adductors",  # тонкая мышца — приводящие мышцы бедра
    "CORB": "shoulder_flexors",  # подключично-лопаточная — сгибатели плечевого сустава
    "TMAJ": "shoulder_extensors",  # большая круглая — разгибатели плечевого сустава
    "TRIlong": "elbow_extensors",  # трёхглавая плеча длинная головка — разгибатели локтевого сустава
    "SUPSP": "shoulder_abductors",  # надостная — отводящие мышцы плеча
    "INFSP": "rotator_cuff",  # подостная — манжета вращателей плеча
    "SUBSC": "rotator_cuff",  # подлопаточная — манжета вращателей плеча
    "TMIN": "rotator_cuff",  # малая круглая — манжета вращателей плеча
    "BIClong": "elbow_flexors",  # двуглавая плеча длинная головка — сгибатели локтевого сустава
    "BICshort": "elbow_flexors",  # двуглавая плеча короткая головка — сгибатели локтевого сустава
    "BRA": "elbow_flexors",  # плечевая — сгибатели локтевого сустава
    "BRD": "elbow_flexors",  # лучевая — сгибатели локтевого сустава
    "TRIlat": "elbow_extensors",  # трёхглавая латеральная головка — разгибатели локтевого сустава
    "TRImed": "elbow_extensors",  # трёхглавая медиальная головка — разгибатели локтевого сустава
    "ANC": "elbow_extensors",  # локтевая — разгибатели локтевого сустава
    "SUP": "pronators_supinators",  # круглый пронатор — пронация и супинация предплечья
}

# --- Functional: конечности (фасции торса — FUNCTIONAL_TORSO_REGEX_MAP) ---
FUNCTIONAL_LIMB_MAP: Dict[str, str] = {
    "psoas": "hip_flexors",  # поясничный мышечный столб — сгибатели бедра в тазобедренном суставе
    "iliacus": "hip_flexors",  # подвздошная — сгибатели бедра в тазобедренном суставе
    "recfem": "hip_flexors",  # прямая мышца бедра — сгибатели бедра в тазобедренном суставе
    "sart": "hip_flexors",  # портняжная — сгибатели бедра в тазобедренном суставе
    "bflh": "hip_extensors",  # двуглавая длинная головка — разгибатели бедра в тазобедренном суставе
    "semimem": "hip_extensors",  # полуперепончатая — разгибатели бедра в тазобедренном суставе
    "semiten": "hip_extensors",  # полусухожильная — разгибатели бедра в тазобедренном суставе
    "addbrev": "hip_adductors",  # короткая приводящая — приводящие мышцы бедра
    "addlong": "hip_adductors",  # длинная приводящая — приводящие мышцы бедра
    "addmagDist": "hip_adductors",  # приводящая большая дистальная часть — приводящие мышцы бедра
    "addmagIsch": "hip_adductors",  # приводящая большая седалищная часть — приводящие мышцы бедра
    "addmagMid": "hip_adductors",  # приводящая большая средняя часть — приводящие мышцы бедра
    "addmagProx": "hip_adductors",  # приводящая большая проксимальная часть — приводящие мышцы бедра
    "grac": "hip_adductors",  # тонкая — приводящие мышцы бедра
    "tfl": "hip_abductors",  # широкая фасция бедра / тракт — отводящие мышцы бедра
    "piri": "hip_abductors",  # грушевидная — отводящие мышцы бедра
    "vaslat": "knee_extensors",  # латеральная широкая бедра — разгибатели коленного сустава
    "vasmed": "knee_extensors",  # медиальная широкая бедра — разгибатели коленного сустава
    "vasint": "knee_extensors",  # промежуточная широкая бедра — разгибатели коленного сустава
    "bfsh": "knee_flexors",  # двуглавая короткая головка — сгибатели коленного сустава
    "gaslat": "knee_flexors",  # икроножная латеральная (здесь как сгибатель колена) — сгибатели коленного сустава
    "gasmed": "knee_flexors",  # икроножная медиальная (здесь как сгибатель колена) — сгибатели коленного сустава
    "soleus": "ankle_plantarflexors",  # камбаловидная — подошвенные сгибатели голеностопа
    "tibpost": "ankle_plantarflexors",  # задняя большеберцовая — подошвенные сгибатели голеностопа
    "perlong": "ankle_plantarflexors",  # длинная малоберцовая — подошвенные сгибатели голеностопа
    "perbrev": "ankle_plantarflexors",  # короткая малоберцовая — подошвенные сгибатели голеностопа
    "fdl": "ankle_plantarflexors",  # длинный сгибатель пальцев стопы — подошвенные сгибатели голеностопа
    "fhl": "ankle_plantarflexors",  # длинный сгибатель большого пальца стопы — подошвенные сгибатели голеностопа
    "tibant": "ankle_dorsiflexors",  # передняя большеберцовая — тыльные разгибатели голеностопа
    "edl": "ankle_dorsiflexors",  # длинный разгибатель пальцев — тыльные разгибатели голеностопа
    "ehl": "ankle_dorsiflexors",  # длинный разгибатель большого пальца стопы — тыльные разгибатели голеностопа
    "CORB": "shoulder_flexors",  # подключично-лопаточная — сгибатели плечевого сустава
    "TMAJ": "shoulder_extensors",  # большая круглая — разгибатели плечевого сустава
    "SUPSP": "shoulder_abductors",  # надостная — отводящие мышцы плеча
    "INFSP": "rotator_cuff",  # подостная — манжета вращателей плеча
    "SUBSC": "rotator_cuff",  # подлопаточная — манжета вращателей плеча
    "TMIN": "rotator_cuff",  # малая круглая — манжета вращателей плеча
    "BIClong": "elbow_flexors",  # двуглавая длинная головка — сгибатели локтевого сустава
    "BICshort": "elbow_flexors",  # двуглавая короткая головка — сгибатели локтевого сустава
    "BRA": "elbow_flexors",  # плечевая — сгибатели локтевого сустава
    "BRD": "elbow_flexors",  # лучевая — сгибатели локтевого сустава
    "TRIlong": "elbow_extensors",  # трёхглавая длинная головка — разгибатели локтевого сустава
    "TRIlat": "elbow_extensors",  # трёхглавая латеральная головка — разгибатели локтевого сустава
    "TRImed": "elbow_extensors",  # трёхглавая медиальная головка — разгибатели локтевого сустава
    "ANC": "elbow_extensors",  # локтевая — разгибатели локтевого сустава
    "SUP": "pronators_supinators",  # круглый пронатор — пронация и супинация предплечья
}

# ============================================================
#  Канонические гранулы — жёсткий маппинг granule → muscle bases
# ============================================================

# Общие определения фасций торса (anatomical + hybrid)
_TORSO_FASCICLE_CANONICAL: Dict[str, List[str]] = {
    "pectoralis": ["PECM1", "PECM2", "PECM3"],
    "deltoids": ["DELT1", "DELT2", "DELT3"],
    "latissimus": ["LAT1", "LAT2", "LAT3"],
    "gluteus_max": ["glmax1", "glmax2", "glmax3"],
    "gluteus_med": ["glmed1", "glmed2", "glmed3"],
    "gluteus_min": ["glmin1", "glmin2", "glmin3"],
    "iliocostalis": [
        "IL_L1", "IL_L2", "IL_L3", "IL_L4",
        "IL_R10", "IL_R11", "IL_R12",
        "IL_R5", "IL_R6", "IL_R7", "IL_R8", "IL_R9",
    ],
    "longissimus": [
        "LTpL_L1", "LTpL_L2", "LTpL_L3", "LTpL_L4", "LTpL_L5",
        "LTpT_R10", "LTpT_R11", "LTpT_R12",
        "LTpT_R4", "LTpT_R5", "LTpT_R6", "LTpT_R7", "LTpT_R8", "LTpT_R9",
        "LTpT_T1", "LTpT_T10", "LTpT_T11", "LTpT_T12",
        "LTpT_T2", "LTpT_T3", "LTpT_T4", "LTpT_T5",
        "LTpT_T6", "LTpT_T7", "LTpT_T8", "LTpT_T9",
    ],
    "multifidus": [
        "MF_m1.laminar", "MF_m1s", "MF_m1t.1", "MF_m1t.2", "MF_m1t.3",
        "MF_m2.laminar", "MF_m2s", "MF_m2t.1", "MF_m2t.2", "MF_m2t.3",
        "MF_m3.laminar", "MF_m3s", "MF_m3t.1", "MF_m3t.2", "MF_m3t.3",
        "MF_m4.laminar", "MF_m4s", "MF_m4t.1", "MF_m4t.2", "MF_m4t.3",
        "MF_m5.laminar", "MF_m5s", "MF_m5t.1", "MF_m5t.2", "MF_m5t.3",
    ],
    "quadratus_lumborum": [
        "QL_ant_I.2-12.1", "QL_ant_I.2-T12",
        "QL_ant_I.3-12.1", "QL_ant_I.3-12.2", "QL_ant_I.3-12.3", "QL_ant_I.3-T12",
        "QL_mid_L2-12.1", "QL_mid_L3-12.1", "QL_mid_L3-12.2", "QL_mid_L3-12.3",
        "QL_mid_L4-12.3",
        "QL_post_I.1-L3", "QL_post_I.2-L2", "QL_post_I.2-L3", "QL_post_I.2-L4",
        "QL_post_I.3-L1", "QL_post_I.3-L2", "QL_post_I.3-L3",
    ],
    "external_oblique": ["EO1", "EO2", "EO3", "EO4", "EO5", "EO6"],
    "internal_oblique": ["IO1", "IO2", "IO3", "IO4", "IO5", "IO6"],
    "rectus_abdominis": ["rect_abd"],
    "paraspinal": [
        "Ps_L1_L2_IVD", "Ps_L1_TP", "Ps_L1_VB",
        "Ps_L2_L3_IVD", "Ps_L2_TP",
        "Ps_L3_L4_IVD", "Ps_L3_TP",
        "Ps_L4_L5_IVD", "Ps_L4_TP",
        "Ps_L5_TP", "Ps_L5_VB",
    ],
    "erector_spinae_cervical": ["ercspn"],
    "internal_oblique_simple": ["intobl"],
    "external_oblique_simple": ["extobl"],
}

# Канон функциональных групп конечностей (hybrid + functional)
_LIMB_FUNCTIONAL_CANONICAL: Dict[str, List[str]] = {
    "hip_flexors": ["iliacus", "psoas", "recfem", "sart"],
    "hip_adductors": [
        "addbrev", "addlong", "addmagDist", "addmagIsch",
        "addmagMid", "addmagProx", "grac",
    ],
    "hip_abductors": ["piri", "tfl"],
    "knee_extensors": ["vasint", "vaslat", "vasmed"],
    "knee_flexors": ["bflh", "bfsh", "semimem", "semiten"],
    "ankle_plantarflexors": [
        "fdl", "fhl", "gaslat", "gasmed",
        "perbrev", "perlong", "soleus", "tibpost",
    ],
    "ankle_dorsiflexors": ["edl", "ehl", "tibant"],
    "shoulder_flexors": ["CORB"],
    "shoulder_extensors": ["TMAJ"],
    "shoulder_abductors": ["SUPSP"],
    "rotator_cuff": ["INFSP", "SUBSC", "TMIN"],
    "elbow_flexors": ["BIClong", "BICshort", "BRA", "BRD"],
    "elbow_extensors": ["ANC", "TRIlat", "TRIlong", "TRImed"],
    "pronators_supinators": ["SUP"],
}

# --- Anatomical (только фасции, конечности — синглтоны) ---
CANONICAL_GRANULES_ANATOMICAL: Dict[str, List[str]] = {
    **_TORSO_FASCICLE_CANONICAL,
}

# --- Functional (всё по функции сустава) ---
CANONICAL_GRANULES_FUNCTIONAL: Dict[str, List[str]] = {
    # Торс: 2 большие функциональные группы
    "spine_flexors": [
        "EO1", "EO2", "EO3", "EO4", "EO5", "EO6",
        "IO1", "IO2", "IO3", "IO4", "IO5", "IO6",
        "extobl", "intobl", "rect_abd",
    ],
    "spine_extensors": [
        "IL_L1", "IL_L2", "IL_L3", "IL_L4",
        "IL_R10", "IL_R11", "IL_R12",
        "IL_R5", "IL_R6", "IL_R7", "IL_R8", "IL_R9",
        "LTpL_L1", "LTpL_L2", "LTpL_L3", "LTpL_L4", "LTpL_L5",
        "LTpT_R10", "LTpT_R11", "LTpT_R12",
        "LTpT_R4", "LTpT_R5", "LTpT_R6", "LTpT_R7", "LTpT_R8", "LTpT_R9",
        "LTpT_T1", "LTpT_T10", "LTpT_T11", "LTpT_T12",
        "LTpT_T2", "LTpT_T3", "LTpT_T4", "LTpT_T5",
        "LTpT_T6", "LTpT_T7", "LTpT_T8", "LTpT_T9",
        "MF_m1.laminar", "MF_m1s", "MF_m1t.1", "MF_m1t.2", "MF_m1t.3",
        "MF_m2.laminar", "MF_m2s", "MF_m2t.1", "MF_m2t.2", "MF_m2t.3",
        "MF_m3.laminar", "MF_m3s", "MF_m3t.1", "MF_m3t.2", "MF_m3t.3",
        "MF_m4.laminar", "MF_m4s", "MF_m4t.1", "MF_m4t.2", "MF_m4t.3",
        "MF_m5.laminar", "MF_m5s", "MF_m5t.1", "MF_m5t.2", "MF_m5t.3",
        "Ps_L1_L2_IVD", "Ps_L1_TP", "Ps_L1_VB",
        "Ps_L2_L3_IVD", "Ps_L2_TP",
        "Ps_L3_L4_IVD", "Ps_L3_TP",
        "Ps_L4_L5_IVD", "Ps_L4_TP",
        "Ps_L5_TP", "Ps_L5_VB",
        "QL_ant_I.2-12.1", "QL_ant_I.2-T12",
        "QL_ant_I.3-12.1", "QL_ant_I.3-12.2", "QL_ant_I.3-12.3", "QL_ant_I.3-T12",
        "QL_mid_L2-12.1", "QL_mid_L3-12.1", "QL_mid_L3-12.2", "QL_mid_L3-12.3",
        "QL_mid_L4-12.3",
        "QL_post_I.1-L3", "QL_post_I.2-L2", "QL_post_I.2-L3", "QL_post_I.2-L4",
        "QL_post_I.3-L1", "QL_post_I.3-L2", "QL_post_I.3-L3",
        "ercspn",
    ],
    # Конечности: функциональные группы с вмёрженными фасциями
    "hip_flexors": ["iliacus", "psoas", "recfem", "sart"],
    "hip_extensors": ["bflh", "glmax1", "glmax2", "glmax3", "semimem", "semiten"],
    "hip_adductors": [
        "addbrev", "addlong", "addmagDist", "addmagIsch",
        "addmagMid", "addmagProx", "grac",
    ],
    "hip_abductors": [
        "glmed1", "glmed2", "glmed3",
        "glmin1", "glmin2", "glmin3",
        "piri", "tfl",
    ],
    "knee_extensors": ["vasint", "vaslat", "vasmed"],
    "knee_flexors": ["bfsh", "gaslat", "gasmed"],
    "ankle_plantarflexors": ["fdl", "fhl", "perbrev", "perlong", "soleus", "tibpost"],
    "ankle_dorsiflexors": ["edl", "ehl", "tibant"],
    "shoulder_flexors": ["CORB", "DELT1", "PECM1", "PECM2", "PECM3"],
    "shoulder_extensors": ["DELT3", "LAT1", "LAT2", "LAT3", "TMAJ"],
    "shoulder_abductors": ["DELT2", "SUPSP"],
    "rotator_cuff": ["INFSP", "SUBSC", "TMIN"],
    "elbow_flexors": ["BIClong", "BICshort", "BRA", "BRD"],
    "elbow_extensors": ["ANC", "TRIlat", "TRIlong", "TRImed"],
    "pronators_supinators": ["SUP"],
}

# --- Hybrid anatomical torso + functional limbs ---
CANONICAL_GRANULES_HYBRID: Dict[str, List[str]] = {
    **_TORSO_FASCICLE_CANONICAL,
    **_LIMB_FUNCTIONAL_CANONICAL,
}


def get_canonical_granules(strategy: str) -> Dict[str, List[str]]:
    """Возвращает canonical dict для выбранной стратегии грануляции."""
    if strategy == "anatomical":
        return CANONICAL_GRANULES_ANATOMICAL
    elif strategy == "functional":
        return CANONICAL_GRANULES_FUNCTIONAL
    elif strategy == "hybrid":
        return CANONICAL_GRANULES_HYBRID
    else:
        logger.warning(f"Unknown strategy '{strategy}', falling back to hybrid")
        return CANONICAL_GRANULES_HYBRID


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
        self.muscle_names = muscle_names
        self.num_muscles = len(muscle_names)
        
        self.__build_action_signatures()

    @classmethod
    def from_env(cls, env) -> "ActionParser":
        muscle_names = []
        for i in range(env.mj_model.nu):
            muscle_names.append(env.mj_model.actuator(i).name)
        return cls(muscle_names)

    @staticmethod
    def parse_side(name: str) -> Tuple[str, str]:
        """Извлекает base_name и side (r/l/c)."""
        if name.endswith("_r"):
            return name[:-2], "r"
        if name.endswith("_l"):
            return name[:-2], "l"
        return name, "c"

    def __build_action_signatures(self) -> None:
        self.action_signatures = []
        for muscle in self.muscle_names:
            base, side = self.parse_side(muscle)
            self.action_signatures.append((base, side, "muscle", "activation"))
        return None

    def get_muscle_grouping(self, strategy: str = "anatomical") -> MuscleGrouping:
        """
        Грануляция мышц по выбранной стратегии.

        Args:
            strategy: "anatomical" | "functional" | "hybrid"
        """
        if strategy == "anatomical":
            return self.granulate(
                ANATOMICAL_TORSO_REGEX_MAP, {},
                strategy, singleton_fallback=True,
            )
        elif strategy == "functional":
            return self.granulate(
                FUNCTIONAL_TORSO_REGEX_MAP, FUNCTIONAL_LIMB_MAP,
                strategy,
            )
        elif strategy == "hybrid":
            return self.granulate(
                ANATOMICAL_TORSO_REGEX_MAP, HYBRID_LIMB_MAP,
                strategy,
            )
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Supported: 'anatomical', 'functional', 'hybrid'."
        )

    def assign_group(
        self,
        muscle_name: str,
        torso_regex_map: List[Tuple[str, str]],
        limb_map: Dict[str, str],
        singleton_fallback: bool = False,
    ) -> str:
        """Определяет group_id для одной мышцы."""
        base, side = self.parse_side(muscle_name)
        side_suffix = f"_{side}" if side != "c" else ""

        # Regex matching (fascicle/functional torso regex)
        for pattern, group_base in torso_regex_map:
            if re.match(pattern, muscle_name):
                return group_base + side_suffix

        # Dict matching (individual limb muscles)
        for key, group_base in limb_map.items():
            if base == key or base.startswith(key):
                return group_base + side_suffix

        if singleton_fallback:
            return base + side_suffix

        return f"other{side_suffix}"

    def granulate(
        self,
        torso_regex_map: List[Tuple[str, str]],
        limb_map: Dict[str, str],
        strategy_name: str,
        singleton_fallback: bool = False,
    ) -> MuscleGrouping:
        """Общий метод грануляции."""
        canonical = get_canonical_granules(strategy_name)

        groups: Dict[str, List[str]] = {}
        muscle_to_group: Dict[str, str] = {}
        granule_base_map: Dict[str, str] = {}
        group_order: List[str] = []

        for muscle in self.muscle_names:
            group = self.assign_group(
                muscle_name=muscle,
                torso_regex_map=torso_regex_map,
                limb_map=limb_map,
                singleton_fallback=singleton_fallback,
            )
            muscle_to_group[muscle] = group
            groups.setdefault(group, []).append(muscle)

        group_order = sorted(groups.keys())

        for group_id in group_order:
            g_base = MuscleGrouping.granule_base_from_group_id(group_id)
            granule_base_map[group_id] = g_base
            if g_base not in canonical and not singleton_fallback:
                logger.warning(
                    f"[{strategy_name}] group '{group_id}' (granule='{g_base}') "
                    f"has no canonical definition. Muscles: {groups[group_id]}"
                )

        logger.info(
            f"{strategy_name}: {len(self.muscle_names)} muscles -> "
            f"{len(group_order)} groups"
        )
        return MuscleGrouping(
            groups=groups,
            group_order=group_order,
            muscle_to_group=muscle_to_group,
            granule_base_map=granule_base_map,
        )
