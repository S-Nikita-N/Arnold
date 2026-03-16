"""
Action-side модули: парсинг мышц, грануляция, sparse attention.
"""

from arnold.action.muscle_parser import (
    parse_muscles_from_xml,
    parse_muscles_myohuman,
)
from arnold.action.muscle_granulator import (
    MuscleGrouping,
    granulate,
    granulate_anatomical,
    granulate_functional,
)
from arnold.action.attention_mask_config import (
    BlockMaskConfig,
    MaskType,
    build_attention_mask,
    get_default_block_configs,
)

__all__ = [
    "parse_muscles_from_xml",
    "parse_muscles_myohuman",
    "MuscleGrouping",
    "granulate",
    "granulate_anatomical",
    "granulate_functional",
    "BlockMaskConfig",
    "MaskType",
    "build_attention_mask",
    "get_default_block_configs",
]
