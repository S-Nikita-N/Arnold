"""
Action Tokenizer — обратная проекция group-level embeddings в per-muscle actions.

Зеркало BodyTokenizer: вместо сжатия наблюдений в группы,
раскрывает group embeddings из decoder в индивидуальные muscle activations.

Поддерживает multi-expert режим: expansion heads шарятся по (group_id, n_muscles),
а index-буферы (reorder) хранятся отдельно для каждого эксперта.

Архитектура A: group queries → decoder → group_out [B,G,D] → ActionTokenizer → [B,A]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import logging

from arnold.action_parser import MuscleGrouping

logger = logging.getLogger(__name__)


class ActionTokenizer(nn.Module):
    """
    Раскрывает group-level decoder output в per-muscle activations.

    Для каждой группы мышц создаётся Linear(embed_dim, n_muscles_in_group).
    Heads шарятся между экспертами, если group_id и размер группы совпадают.
    Reorder-буферы per-expert: маппят порядок групп → оригинальный порядок мышц.
    """

    def __init__(
        self,
        groupings: Dict[str, MuscleGrouping],
        action_signatures: Dict[str, List[Tuple[str, ...]]],
        embed_dim: int,
    ):
        """
        Args:
            groupings: expert_name → MuscleGrouping
            action_signatures: expert_name → list of (base, side, "muscle", "activation") tuples
            embed_dim: размерность embeddings (должна совпадать с decoder output)
        """
        super().__init__()
        self.embed_dim = embed_dim

        self.expert_names: List[str] = sorted(groupings.keys())
        self._expert_group_order: Dict[str, List[str]] = {}
        self._expert_group_signatures: Dict[str, List[Tuple[str, ...]]] = {}
        self._expert_group_sizes: Dict[str, List[int]] = {}

        # Собираем все уникальные (group_id, n_muscles) для shared heads
        head_configs: Dict[str, int] = {}  # key → n_muscles

        for expert_name in self.expert_names:
            grouping = groupings[expert_name]
            act_sigs = action_signatures[expert_name]

            group_order = grouping.group_order
            self._expert_group_order[expert_name] = group_order

            # Group signatures: берём сигнатуру первой мышцы в каждой группе
            muscle_name_to_sig = {}
            for sig in act_sigs:
                # sig = (base, side, "muscle", "activation")
                base, side = sig[0], sig[1]
                if side == "c":
                    muscle_name_to_sig[base] = sig
                else:
                    muscle_name_to_sig[f"{base}_{side}"] = sig

            group_sigs = []
            group_sizes = []
            for group_id in group_order:
                muscles = grouping.groups[group_id]
                group_sizes.append(len(muscles))

                # Берём сигнатуру первой мышцы как representative
                first_muscle = muscles[0]
                if first_muscle in muscle_name_to_sig:
                    group_sigs.append(muscle_name_to_sig[first_muscle])
                else:
                    # Fallback: парсим имя мышцы
                    if first_muscle.endswith("_r"):
                        group_sigs.append((first_muscle[:-2], "r", "muscle", "activation"))
                    elif first_muscle.endswith("_l"):
                        group_sigs.append((first_muscle[:-2], "l", "muscle", "activation"))
                    else:
                        group_sigs.append((first_muscle, "c", "muscle", "activation"))

            self._expert_group_signatures[expert_name] = group_sigs
            self._expert_group_sizes[expert_name] = group_sizes

            # Регистрируем head configs
            for group_id, n_muscles in zip(group_order, group_sizes):
                key = self._head_key(group_id, n_muscles)
                if key in head_configs:
                    assert head_configs[key] == n_muscles, (
                        f"Head size mismatch for {key}: {head_configs[key]} vs {n_muscles}"
                    )
                else:
                    head_configs[key] = n_muscles

            # Reorder index: маппит grouped order → original muscle order
            # grouped_muscles — конкатенация мышц по group_order
            grouped_muscles: List[str] = []
            for group_id in group_order:
                grouped_muscles.extend(grouping.groups[group_id])

            # original_muscles — порядок из action_signatures
            original_muscles: List[str] = []
            for sig in act_sigs:
                base, side = sig[0], sig[1]
                if side == "c":
                    original_muscles.append(base)
                else:
                    original_muscles.append(f"{base}_{side}")

            # reorder_idx: для каждой позиции в original, какая позиция в grouped
            muscle_to_grouped_pos = {m: i for i, m in enumerate(grouped_muscles)}
            reorder_idx = []
            for m in original_muscles:
                if m in muscle_to_grouped_pos:
                    reorder_idx.append(muscle_to_grouped_pos[m])
                else:
                    logger.warning(
                        f"Muscle '{m}' not found in any group for expert '{expert_name}'. "
                        f"This may cause incorrect action ordering."
                    )
                    reorder_idx.append(0)

            self.register_buffer(
                f"reorder_idx_{expert_name}",
                torch.tensor(reorder_idx, dtype=torch.long),
            )

        # Создаём shared expansion heads
        self.expand_heads = nn.ModuleDict()
        for key, n_muscles in head_configs.items():
            self.expand_heads[key] = nn.Linear(embed_dim, n_muscles)

        # Инициализация весов
        for head in self.expand_heads.values():
            nn.init.orthogonal_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)

        # Логгирование
        for expert_name in self.expert_names:
            grouping = groupings[expert_name]
            sizes = self._expert_group_sizes[expert_name]
            logger.info(
                f"ActionTokenizer [{expert_name}]: "
                f"{grouping.n_muscles} muscles -> {grouping.n_groups} groups "
                f"(sizes: min={min(sizes)}, max={max(sizes)}, "
                f"mean={sum(sizes) / len(sizes):.1f})"
            )
            # Предупреждение о мышцах в "other" группе
            for group_id in grouping.group_order:
                if group_id.startswith("other"):
                    n = len(grouping.groups[group_id])
                    logger.warning(
                        f"  [{expert_name}] {n} muscles in '{group_id}' group "
                        f"(ungrouped): {grouping.groups[group_id][:5]}..."
                    )

    @staticmethod
    def _head_key(group_id: str, n_muscles: int) -> str:
        """Ключ для shared expansion head."""
        # Sanitize group_id для nn.ModuleDict (не допускает точки)
        safe_id = group_id.replace(".", "_").replace("-", "_")
        return f"{safe_id}__{n_muscles}"

    def get_group_signatures(self, expert_name: str) -> List[Tuple[str, ...]]:
        """Возвращает group-level signatures для vocab embedding lookup."""
        return self._expert_group_signatures[expert_name]

    def get_n_groups(self, expert_name: str) -> int:
        """Количество групп для эксперта."""
        return len(self._expert_group_order[expert_name])

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
        group_order = self._expert_group_order[expert_name]
        group_sizes = self._expert_group_sizes[expert_name]

        mean_parts: List[torch.Tensor] = []
        hidden_parts: List[torch.Tensor] = []

        for g_idx, (group_id, n_muscles) in enumerate(zip(group_order, group_sizes)):
            g_emb = group_out[:, g_idx, :]  # [B, D]

            # Expansion head: [B, D] → [B, n_muscles]
            key = self._head_key(group_id, n_muscles)
            muscle_means = self.expand_heads[key](g_emb)
            mean_parts.append(muscle_means)

            # Hidden: per-muscle hidden states для covariance.
            # g_emb после LayerNorm: ||g_emb|| ≈ √D.
            # W_norm — единичные направления ролей мышц в группе.
            # g_emb ⊙ W_norm даёт ||hidden|| ≈ 1, а в оригинальном пути
            # ||action_out|| ≈ √D. Масштабируем на √D для совместимости с cov_factor.
            W_norm = F.normalize(self.expand_heads[key].weight, dim=-1)  # [n_muscles, D]
            muscle_h = g_emb.unsqueeze(1) * W_norm.unsqueeze(0)  # [B, n_muscles, D]
            hidden_parts.append(muscle_h * (D ** 0.5))

        # Concatenate в grouped order
        mean_grouped = torch.cat(mean_parts, dim=1)     # [B, A_total]
        hidden_grouped = torch.cat(hidden_parts, dim=1)  # [B, A_total, D]

        # Reorder в оригинальный порядок мышц
        reorder_idx = getattr(self, f"reorder_idx_{expert_name}")  # [A]

        mean = torch.gather(
            mean_grouped, 1,
            reorder_idx.unsqueeze(0).expand(B, -1),
        )  # [B, A]

        hidden = torch.gather(
            hidden_grouped, 1,
            reorder_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, D),
        )  # [B, A, D]

        return mean, hidden
