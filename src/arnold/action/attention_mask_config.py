"""
Конфигурация масок attention для action decoder.

Поддерживает гибкую настройку для каждого decoder block отдельно.
Совмещает подходы B (hierarchical) и C (block-sparse) из README.

Типы масок:
- "full": полное внимание (как обычный transformer)
- "block_diagonal": только внутри групп мышц (блоки по группам)
- "hierarchical": group tokens + muscles; группы смотрят друг на друга,
  мышцы смотрят на свою группу и мышцы своей группы
- "group_only": только group tokens (для Architecture A)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import torch

import logging

from arnold.action.muscle_granulator import MuscleGrouping

logger = logging.getLogger(__name__)


class MaskType(str, Enum):
    FULL = "full"
    BLOCK_DIAGONAL = "block_diagonal"
    HIERARCHICAL = "hierarchical"
    GROUP_ONLY = "group_only"


@dataclass
class BlockMaskConfig:
    """Конфигурация маски для одного decoder block."""
    mask_type: MaskType = MaskType.FULL
    # Для hierarchical: включать group tokens в sequence
    use_group_tokens: bool = False
    # Group tokens могут смотреть на все группы (True) или только на свои мышцы (False)
    groups_attend_all_groups: bool = True
    # Muscles смотрят на group token своей группы
    muscles_attend_group: bool = True
    # Muscles смотрят на muscles своей группы
    muscles_attend_same_group: bool = True


def build_attention_mask(
    grouping: MuscleGrouping,
    muscle_names: List[str],
    config: BlockMaskConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """
    Строит attention mask для decoder.

    Mask[i,j] = 0 где внимание разрешено, -inf где запрещено.
    Размер: [seq_len, seq_len] для использования в scaled_dot_product_attention.

    Args:
        grouping: результат грануляции
        muscle_names: порядок мышц (соответствует action_signatures)
        config: конфигурация маски
        device, dtype: для тензора

    Returns:
        attn_mask [seq_len, seq_len] или None для full attention
    """
    n_muscles = len(muscle_names)
    muscle_to_idx = {m: i for i, m in enumerate(muscle_names)}

    if config.mask_type == MaskType.FULL:
        return None

    if config.mask_type == MaskType.GROUP_ONLY:
        n = grouping.n_groups
        return None  # Group-only decoder обрабатывается отдельно

    if config.mask_type == MaskType.BLOCK_DIAGONAL:
        mask = torch.zeros(n_muscles, n_muscles, device=device, dtype=dtype)
        mask.fill_(float("-inf"))
        for group_id, muscles in grouping.groups.items():
            indices = [muscle_to_idx[m] for m in muscles if m in muscle_to_idx]
            for i in indices:
                for j in indices:
                    mask[i, j] = 0
        return mask

    if config.mask_type == MaskType.HIERARCHICAL:
        n_groups = grouping.n_groups
        n_total = n_groups + n_muscles
        mask = torch.zeros(n_total, n_total, device=device, dtype=dtype)
        mask.fill_(float("-inf"))

        # Group tokens: 0 .. n_groups-1
        # Muscle tokens: n_groups .. n_total-1
        group_to_offset = {g: grouping.get_group_index(g) for g in grouping.group_order}
        muscle_to_seq_idx = {m: n_groups + muscle_to_idx[m] for m in muscle_names if m in muscle_to_idx}

        # 1. Group-to-group: все смотрят на все (если groups_attend_all_groups)
        if config.groups_attend_all_groups:
            mask[:n_groups, :n_groups] = 0

        # 2. Group-to-muscle: группа g смотрит на свои мышцы
        for group_id, muscles in grouping.groups.items():
            g_idx = group_to_offset.get(group_id, -1)
            if g_idx < 0:
                continue
            for m in muscles:
                if m in muscle_to_seq_idx:
                    mask[g_idx, muscle_to_seq_idx[m]] = 0

        # 3. Muscle-to-group: мышца смотрит на свою группу
        if config.muscles_attend_group:
            for m, g_id in grouping.muscle_to_group.items():
                if m not in muscle_to_seq_idx:
                    continue
                g_idx = group_to_offset.get(g_id, -1)
                if g_idx >= 0:
                    mask[muscle_to_seq_idx[m], g_idx] = 0

        # 4. Muscle-to-muscle: только внутри группы
        if config.muscles_attend_same_group:
            for group_id, muscles in grouping.groups.items():
                indices = [muscle_to_seq_idx[m] for m in muscles if m in muscle_to_seq_idx]
                for i in indices:
                    for j in indices:
                        mask[i, j] = 0

        return mask

    return None


def get_default_block_configs(
    num_layers: int,
    strategy: str = "progressive",
) -> List[BlockMaskConfig]:
    """
    Дефолтные конфиги для каждого decoder block.

    strategy:
    - "full": все блоки full attention
    - "sparse_all": все блоки block_diagonal
    - "progressive": первые блоки sparse, последние full (или наоборот)
    - "alternating": чередование full и sparse
    """
    configs = []
    for i in range(num_layers):
        if strategy == "full":
            configs.append(BlockMaskConfig(mask_type=MaskType.FULL))
        elif strategy == "sparse_all":
            configs.append(BlockMaskConfig(mask_type=MaskType.BLOCK_DIAGONAL))
        elif strategy == "hierarchical_all":
            configs.append(BlockMaskConfig(
                mask_type=MaskType.HIERARCHICAL,
                use_group_tokens=True,
                groups_attend_all_groups=True,
                muscles_attend_group=True,
                muscles_attend_same_group=True,
            ))
        elif strategy == "progressive":
            # Первая половина - sparse, вторая - full (для refinement)
            if i < num_layers // 2:
                configs.append(BlockMaskConfig(mask_type=MaskType.BLOCK_DIAGONAL))
            else:
                configs.append(BlockMaskConfig(mask_type=MaskType.FULL))
        elif strategy == "alternating":
            configs.append(BlockMaskConfig(
                mask_type=MaskType.BLOCK_DIAGONAL if i % 2 == 0 else MaskType.FULL
            ))
        else:
            configs.append(BlockMaskConfig(mask_type=MaskType.FULL))
    return configs
