"""
Body Tokenizer — универсальная проекция наблюдений в эмбеддинги.

Принимает список BodyGroup (сгенерированных с любой гранулярностью)
и проецирует каждую группу через Linear(n_features * history_len, embed_dim).
Группы одного type разделяют один проектор.

Гранулярность управляется в ObservationParser.get_body_groups(granularity):
  "scalar"   → ~1583 токенов, Linear(5, 128) — как SensoryEncoder
  "per_spec" → ~500 токенов, по (тело, тип_obs)
  "per_body" → ~112 токенов, всё тело в одном токене
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from arnold.observation_parser import BodyGroup


class BodyTokenizer(nn.Module):
    """
    Преобразует flat observations [B, n_obs, history_len] в токены [B, n_groups, embed_dim].

    Для каждого type создаётся отдельный Linear проектор (группы
    одного типа имеют одинаковую входную размерность).
    Количество и состав групп определяется granularity в ObservationParser.
    """

    def __init__(
        self,
        groups: List[BodyGroup],
        history_len: int,
        embed_dim: int,
    ):
        super().__init__()
        self.history_len = history_len
        self.embed_dim = embed_dim

        sorted_groups = sorted(groups, key=lambda g: (g.type, g.name, g.side))
        self.group_signatures: List[Tuple[str, ...]] = [g.signature for g in sorted_groups]
        self.n_groups = len(sorted_groups)

        type_to_indices: Dict[str, List[List[int]]] = {}
        for g in sorted_groups:
            type_to_indices.setdefault(g.type, []).append(g.indices)

        self.type_order = sorted(type_to_indices.keys())

        self.projectors = nn.ModuleDict()
        for gtype in self.type_order:
            idx_lists = type_to_indices[gtype]
            n_features = len(idx_lists[0])
            input_dim = n_features * history_len
            self.projectors[gtype] = nn.Linear(input_dim, embed_dim)
            idx_tensor = torch.tensor(idx_lists, dtype=torch.long)
            self.register_buffer(f"idx_{gtype}", idx_tensor)

    def forward(self, obs_timeseries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_timeseries: [B, n_obs_flat, history_len] — нормализованные наблюдения

        Returns:
            [B, n_groups, embed_dim] — body-level токены
        """
        B = obs_timeseries.shape[0]
        parts = []
        for gtype in self.type_order:
            idx = getattr(self, f"idx_{gtype}")  # [n_groups_of_type, n_features]
            n_g, n_f = idx.shape
            gathered = obs_timeseries[:, idx, :]  # [B, n_g, n_f, H]
            gathered = gathered.reshape(B, n_g, n_f * self.history_len)
            parts.append(self.projectors[gtype](gathered))
        return torch.cat(parts, dim=1)  # [B, n_groups, embed_dim]
