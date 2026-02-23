"""
Body Tokenizer — группировка наблюдений по телам для body-level tokenization.

Заменяет SensoryEncoder. Вместо ~953 скалярных токенов (по одному на каждый
элемент наблюдения) создаёт ~70 body-level токенов, объединяя все наблюдения
одного тела в один вектор и проецируя через Linear.

Ускорение attention: 953² → 70² = ~185x.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from arnold.observation_parser import BodyGroup


class BodyTokenizer(nn.Module):
    """
    Преобразует flat observations [B, n_obs, history_len] в body-level токены
    [B, n_groups, embed_dim].

    Для каждого типа группы (root, body, muscle, task, task_muscle, feet)
    создаётся отдельный Linear проектор, т.к. разные типы имеют разную
    входную размерность.
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

        sorted_groups = sorted(groups, key=lambda g: (g.group_type, g.name, g.side))
        self.group_signatures: List[Tuple[str, ...]] = [g.signature for g in sorted_groups]
        self.n_groups = len(sorted_groups)

        type_to_indices: Dict[str, List[List[int]]] = {}
        for g in sorted_groups:
            type_to_indices.setdefault(g.group_type, []).append(g.flat_indices)

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
