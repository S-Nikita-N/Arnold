"""
Action Tokenizer — обратная проекция group-level embeddings в per-muscle actions.

Канонические гранулы (canonical_granules):
- Каждая гранула имеет ФИКСИРОВАННЫЙ набор мышц и expansion head.
- Expansion heads шарятся между left/right и между экспертами.
- Если эксперт использует подмножество мышц гранулы, head всё равно
  каноничного размера, а нужные позиции отбираются через select_idx.

Vocabulary signatures:
- Каждая group query использует (granule_base, side, "muscle", "activation")
  вместо сигнатуры первой мышцы. Это даёт семантически правильные
  role embeddings из SensorimotorVocabulary.

Архитектура A: group queries → decoder → group_out [B,G,D] → ActionTokenizer → [B,A]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import logging

from arnold.action_parser import MuscleGrouping, get_canonical_granules

logger = logging.getLogger(__name__)


class ActionTokenizer(nn.Module):
    """Раскрывает group-level decoder output в per-muscle activations."""

    def __init__(
        self,
        groupings: Dict[str, MuscleGrouping],
        action_signatures: Dict[str, List[Tuple[str, ...]]],
        embed_dim: int,
        strategy: str = "hybrid",
    ):
        """
        Args:
            groupings: expert_name → MuscleGrouping (с granule_base_map)
            action_signatures: expert_name → list of (base, side, "muscle", "activation") tuples
            embed_dim: размерность embeddings (должна совпадать с decoder output)
            strategy: "anatomical" | "functional" | "hybrid"
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.strategy = strategy
        self.expert_names: List[str] = sorted(groupings.keys())
        self.canonical_granules = get_canonical_granules(strategy)

        # Per-expert metadata: (granule_base, side) tuples instead of string group_ids
        self.expert_groups: Dict[str, List[Tuple[str, str]]] = {}  # [(base, side), ...]
        self.expert_group_sizes: Dict[str, List[int]] = {}

        self.build_grouping(groupings, action_signatures)
        self.log_summary(groupings)

    def build_grouping(
        self,
        groupings: Dict[str, MuscleGrouping],
        action_signatures: Dict[str, List[Tuple[str, ...]]],
    ) -> None:
        """Строит select_idx и reorder_idx буферы для каждого эксперта."""
        head_output_dims: Dict[str, int] = {}

        for expert_name in self.expert_names:
            grouping = groupings[expert_name]
            act_sigs = action_signatures[expert_name]

            group_signatures: List[Tuple[str, str]] = []  # (granule_base, side)
            group_sizes: List[int] = []

            for g_idx, group_id in enumerate(grouping.group_order):
                muscles = grouping.groups[group_id]
                granule_base = grouping.granule_base_map[group_id]
                side = MuscleGrouping.parse_side_from_group_id(group_id)

                group_signatures.append((granule_base, side))
                group_sizes.append(len(muscles))

                # Canonical head size
                if "." in granule_base or "-" in granule_base:
                    raise ValueError(
                        f"granule_base '{granule_base}' contains '.' or '-' which are "
                        f"not allowed as nn.ModuleDict keys. Fix canonical granule naming."
                    )

                if granule_base not in head_output_dims:
                    head_output_dims[granule_base] = (
                        len(self.canonical_granules[granule_base]) 
                        if granule_base in self.canonical_granules
                        else len(muscles)
                    )

                # select_idx: canonical positions → expert muscles
                select_idx = self.build_select_idx(granule_base, muscles)
                self.register_buffer(
                    f"select_{expert_name}_{g_idx}",
                    torch.tensor(select_idx, dtype=torch.long),
                )

            self.expert_groups[expert_name] = group_signatures
            self.expert_group_sizes[expert_name] = group_sizes

            # reorder_idx: grouped order → original muscle order
            reorder_idx = self.build_reorder_idx(grouping, act_sigs)
            self.register_buffer(
                f"reorder_idx_{expert_name}",
                torch.tensor(reorder_idx, dtype=torch.long),
            )

        self.build_output_heads(head_output_dims)

    def build_select_idx(
        self,
        granule_base: str,
        muscles: List[str]
    ) -> List[int]:
        """Строит select_idx: маппинг canonical positions → expert muscles."""
        if granule_base not in self.canonical_granules:
            return list(range(len(muscles)))

        select_idx: List[int] = []
        canonical_muscles = self.canonical_granules[granule_base]
        canonical_pos = {base: idx for idx, base in enumerate(canonical_muscles)}

        for muscle_name in muscles:
            if muscle_name.endswith("_r") or muscle_name.endswith("_l"):
                muscle_name = muscle_name[:-2]
            select_idx.append(canonical_pos[muscle_name])

        return select_idx

    def build_reorder_idx(
        self,
        grouping: MuscleGrouping,
        act_sigs: List[Tuple[str, ...]],
    ) -> List[int]:
        """Строит reorder_idx: grouped order → original muscle order."""
        idx = 0
        reorder_idx: List[int] = []

        muscle_to_grouped_pos: Dict[str, int] = {}
        for group_id in grouping.group_order:
            for muscle in grouping.groups[group_id]:
                muscle_to_grouped_pos[muscle] = idx
                idx += 1

        for sig in act_sigs:
            base, side = sig[0], sig[1]
            muscle_name = base if side == "c" else f"{base}_{side}"
            reorder_idx.append(muscle_to_grouped_pos[muscle_name])

        return reorder_idx

    def build_output_heads(
        self,
        head_output_dims: Dict[str, int],
    ) -> None:
        """Создаёт shared canonical expansion heads."""
        self.expand_heads = nn.ModuleDict()
        for key, output_dim in head_output_dims.items():
            self.expand_heads[key] = nn.Linear(self.embed_dim, output_dim)

        for head in self.expand_heads.values():
            nn.init.orthogonal_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)

    def log_summary(
        self,
        groupings: Dict[str, MuscleGrouping]
    ) -> None:
        """Логгирует статистику по экспертам."""
        for expert_name in self.expert_names:
            grouping = groupings[expert_name]
            sizes = self.expert_group_sizes[expert_name]
            logger.info(
                f"ActionTokenizer [{expert_name}]: "
                f"{grouping.n_muscles} muscles -> {grouping.n_groups} groups "
                f"(sizes: min={min(sizes)}, max={max(sizes)}, "
                f"mean={sum(sizes) / len(sizes):.1f})"
            )
            canonical_sizes = [
                len(self.canonical_granules[base]) if base in self.canonical_granules else sizes[i]
                for i, (base, _side) in enumerate(self.expert_groups[expert_name])
            ]
            logger.info(
                f"  Canonical head sizes: min={min(canonical_sizes)}, "
                f"max={max(canonical_sizes)}, total_params={sum(canonical_sizes) * self.embed_dim}"
            )

    def get_group_signatures(
        self,
        expert_name: str
    ) -> List[Tuple[str, ...]]:
        """Возвращает granule-level signatures для vocab embedding lookup."""
        return [
            (base, side, "muscle", "activation")
            for base, side in self.expert_groups[expert_name]
        ]

    def decode(
        self,
        group_out: torch.Tensor,
        expert_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Декодирует group embeddings в per-muscle actions.

        Args:
            group_out: [B, G, D] — output from action decoder
            expert_name: ключ эксперта

        Returns:
            mean: [B, A] — per-muscle mean activations (в оригинальном порядке мышц)
            muscle_hidden: [B, A, D] — per-muscle hidden states (для covariance)
        """
        B, G, D = group_out.shape
        group_signatures = self.expert_groups[expert_name]
        group_sizes = self.expert_group_sizes[expert_name]

        mean_parts: List[torch.Tensor] = []
        hidden_parts: List[torch.Tensor] = []

        for g_idx, ((granule_base, _side), n_expert_muscles) in enumerate(
            zip(group_signatures, group_sizes)
        ):
            # Project in actions
            head = self.expand_heads[granule_base]
            g_emb = group_out[:, g_idx, :]  # [B, D]
            canonical_means = head(g_emb)

            # Select expert's muscles from canonical positions
            select_idx = getattr(self, f"select_{expert_name}_{g_idx}")
            muscle_means = canonical_means[:, select_idx]  # [B, n_expert_muscles]
            mean_parts.append(muscle_means)

            # Hidden: per-muscle hidden states для covariance
            W_norm = F.normalize(head.weight, dim=-1)  # [n_canonical, D]
            canonical_h = g_emb.unsqueeze(1) * W_norm.unsqueeze(0)  # [B, n_canonical, D]
            muscle_h = canonical_h[:, select_idx]  # [B, n_expert_muscles, D]
            hidden_parts.append(muscle_h * (D ** 0.5))

        # Concatenate в grouped order
        mean_grouped = torch.cat(mean_parts, dim=1)     # [B, A_total]
        hidden_grouped = torch.cat(hidden_parts, dim=1)  # [B, A_total, D]

        # Reorder в оригинальный порядок мышц
        reorder_idx = getattr(self, f"reorder_idx_{expert_name}")

        mean = torch.gather(
            mean_grouped,
            1,
            reorder_idx.unsqueeze(0).expand(B, -1),
        )  # [B, A]

        hidden = torch.gather(
            hidden_grouped, 
            1,
            reorder_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, D),
        )  # [B, A, D]

        return mean, hidden
