"""
Body Tokenizer — универсальная проекция наблюдений в эмбеддинги.

Принимает список BodyGroup (сгенерированных с любой гранулярностью)
и проецирует каждую группу через Linear(n_features * history_len, embed_dim).
Группы одного type разделяют один проектор.

Поддерживает multi-expert режим: projectors шарятся по типу,
а index-буферы хранятся отдельно для каждого эксперта.

Гранулярность управляется в ObservationParser.get_body_groups(granularity):
  "scalar"   → ~1583 токенов, Linear(5, 128) — как SensoryEncoder
  "per_spec" → ~500 токенов, по (тело, тип_obs)
  "per_body" → ~112 токенов, всё тело в одном токене
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union, Optional
from arnold.observation_parser import BodyGroup


class BodyTokenizer(nn.Module):
    """
    Преобразует flat observations [B, n_obs, history_len] в токены [B, n_groups, embed_dim].

    Для каждого type создаётся отдельный Linear проектор (группы
    одного типа имеют одинаковую входную размерность).

    Multi-expert: projectors шарятся между экспертами (одинаковый тип →
    одинаковый проектор), а index-буферы хранятся per-expert.
    При forward(obs, expert_name) используются индексы конкретного эксперта.
    """

    def __init__(
        self,
        groups: Union[List[BodyGroup], Dict[str, List[BodyGroup]]],
        history_len: int,
        embed_dim: int,
    ):
        """
        Args:
            groups: List[BodyGroup] (single-expert, backward compat) или
                    Dict[expert_name → List[BodyGroup]] (multi-expert)
            history_len: длина истории наблюдений
            embed_dim: размерность эмбеддинга
        """
        super().__init__()
        self.history_len = history_len
        self.embed_dim = embed_dim

        groups_by_expert = groups

        self.expert_names: List[str] = sorted(groups_by_expert.keys())

        type_features: Dict[str, int] = {}
        self._expert_group_signatures: Dict[str, List[Tuple[str, ...]]] = {}
        self._expert_type_orders: Dict[str, List[str]] = {}

        for expert_name in self.expert_names:
            expert_groups = groups_by_expert[expert_name]
            sorted_groups = sorted(expert_groups, key=lambda g: (g.type, g.name, g.side))
            self._expert_group_signatures[expert_name] = [g.signature for g in sorted_groups]

            type_to_indices: Dict[str, List[List[int]]] = {}
            for g in sorted_groups:
                type_to_indices.setdefault(g.type, []).append(g.indices)

            type_order = sorted(type_to_indices.keys())
            self._expert_type_orders[expert_name] = type_order

            for gtype in type_order:
                idx_lists = type_to_indices[gtype]
                n_features = len(idx_lists[0])

                if gtype in type_features:
                    if type_features[gtype] != n_features:
                        raise ValueError(
                            f"Feature count mismatch for type '{gtype}': "
                            f"expected {type_features[gtype]}, got {n_features} "
                            f"(expert '{expert_name}'). "
                            f"Experts must share the same observation structure per type."
                        )
                else:
                    type_features[gtype] = n_features

                idx_tensor = torch.tensor(idx_lists, dtype=torch.long)
                self.register_buffer(f"idx_{expert_name}_{gtype}", idx_tensor)

        all_types = sorted(type_features.keys())
        self.projectors = nn.ModuleDict()
        for gtype in all_types:
            n_features = type_features[gtype]
            input_dim = n_features * history_len
            self.projectors[gtype] = nn.Linear(input_dim, embed_dim)

    def forward(
        self,
        obs_timeseries: torch.Tensor,
        expert_name: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Args:
            obs_timeseries: [B, n_obs_flat, history_len]
            expert_name: ключ эксперта (None → default / единственный)

        Returns:
            [B, n_groups, embed_dim]
        """
        B = obs_timeseries.shape[0]
        parts = []
        for gtype in self._expert_type_orders[expert_name]:
            idx = getattr(self, f"idx_{expert_name}_{gtype}")
            n_g, n_f = idx.shape
            gathered = obs_timeseries[:, idx, :]          # [B, n_g, n_f, H]
            gathered = gathered.reshape(B, n_g, n_f * self.history_len)
            parts.append(self.projectors[gtype](gathered))

        return torch.cat(parts, dim=1)                    # [B, n_groups, embed_dim]

    def get_group_signatures(self, expert_name: Optional[str] = None) -> List[Tuple[str, ...]]:
        return self._expert_group_signatures[expert_name]
