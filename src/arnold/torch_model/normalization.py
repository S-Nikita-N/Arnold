"""
Per-signature нормализация наблюдений.
Хранит бегущие статистики (mean/std) для каждой signature.

Векторизованная реализация: Python-цикл при первом вызове строит
индексную карту (signature → позиция в stats-тензорах), дальше
всё через батчевые тензорные операции.
"""

import torch
import torch.nn as nn

from typing import Any


########################################
#         Signature normalizer         #
########################################


class SignatureNormalizerModule(nn.Module):
    """
    Пер-сигнатурная нормализация с бегущими статистиками.

    Статистики хранятся в тензорах для быстрого доступа.
    Python-цикл выполняется только один раз при первом встрече
    набора signatures — дальше работает индексная карта.
    """

    def __init__(self):
        super().__init__()
        self.stats: dict[str, tuple[int, torch.Tensor, torch.Tensor]] = {}
        self._index_cache: dict[
            int,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]],
        ] = {}

    def _apply(self, fn):
        super()._apply(fn)
        for key, (count, mean, M2) in self.stats.items():
            self.stats[key] = (count, fn(mean), fn(M2))
        self._index_cache.clear()
        return self

    @staticmethod
    def _sig_key(sig) -> str:
        if isinstance(sig, (list, tuple)):
            return "|".join(sig)
        return str(sig)

    def get_index_map(
        self,
        signatures: list[tuple[str, ...]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """
        Возвращает закешированную индексную карту для данного набора signatures.

        Returns:
            known_t: [n_known] long — индексы в signatures, для которых есть
                     статистика (count >= 2)
            unknown_t: [n_unknown] long — остальные индексы
            keys: список строковых ключей (n_obs), по одному на signature
        """
        cache_key = id(signatures)
        if cache_key in self._index_cache:
            cached = self._index_cache[cache_key]
            if cached[0].device == device:
                return cached
            moved = (
                cached[0].to(device),
                cached[1].to(device),
                cached[2],
            )
            # Кешируем перенесённые тензоры, чтобы не повторять .to()
            # на каждом вызове
            self._index_cache[cache_key] = moved
            return moved

        keys = [self._sig_key(sig) for sig in signatures]
        known_idx = []
        unknown_idx = []
        for i, key in enumerate(keys):
            if key in self.stats and self.stats[key][0] >= 2:
                known_idx.append(i)
            else:
                unknown_idx.append(i)

        known_t = torch.tensor(known_idx, dtype=torch.long, device=device)
        unknown_t = torch.tensor(unknown_idx, dtype=torch.long, device=device)

        result = (known_t, unknown_t, keys)
        if len(self._index_cache) > 10:
            self._index_cache.clear()
        self._index_cache[cache_key] = result
        return result

    def _invalidate_cache(self):
        self._index_cache.clear()

    def update(self, signatures: list[tuple[str, ...]], values: torch.Tensor):
        """
        Обновляет статистики (Welford's online algorithm), векторизовано.

        Args:
            signatures: длина n_obs
            values: [batch, n_obs, history_len]
        """
        b, n, h = values.shape
        vals = values.detach()[:, :, -1]  # [batch, n_obs]
        batch_means = vals.mean(dim=0)  # [n_obs]
        batch_vars = vals.var(dim=0, unbiased=False)  # [n_obs]
        count_new = b

        cache_invalidated = False
        keys = [self._sig_key(sig) for sig in signatures]
        for i, key in enumerate(keys):
            mean_new = batch_means[i]
            M2_new = batch_vars[i] * count_new
            if key not in self.stats:
                self.stats[key] = (count_new, mean_new, M2_new)
                cache_invalidated = True
            else:
                count, mean, M2 = self.stats[key]
                delta = mean_new - mean
                total = count + count_new
                mean = mean + delta * (count_new / total)
                M2 = M2 + M2_new + delta * delta * count * count_new / total
                self.stats[key] = (total, mean, M2)

        if cache_invalidated:
            self._invalidate_cache()

    def normalize(
        self,
        signatures: list[tuple[str, ...]],
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Нормализует значения используя накопленные статистики.
        Векторизованная реализация — без Python for-loop на hot path.
        """
        b, n, h = values.shape
        device = values.device
        known_t, unknown_t, keys = self.get_index_map(signatures, device)

        if len(unknown_t) == 0 and len(known_t) == n:
            means = torch.stack([self.stats[keys[i]][1] for i in range(n)]).to(
                device,
            )
            M2s = torch.stack([self.stats[keys[i]][2] for i in range(n)]).to(
                device,
            )
            counts = torch.tensor(
                [self.stats[keys[i]][0] for i in range(n)],
                dtype=values.dtype,
                device=device,
            )
            stds = torch.sqrt(M2s / counts + 1e-8)
            return (values - means.view(1, n, 1)) / stds.view(1, n, 1)

        elif len(known_t) == 0:
            return values.clone()

        else:
            out = values.clone()
            k_idx = known_t.tolist()
            means = torch.stack([self.stats[keys[i]][1] for i in k_idx]).to(
                device,
            )
            M2s = torch.stack([self.stats[keys[i]][2] for i in k_idx]).to(
                device,
            )
            counts = torch.tensor(
                [self.stats[keys[i]][0] for i in k_idx],
                dtype=values.dtype,
                device=device,
            )
            stds = torch.sqrt(M2s / counts + 1e-8)
            k = len(k_idx)
            out[:, known_t, :] = (
                values[:, known_t, :] - means.view(1, k, 1)
            ) / stds.view(1, k, 1)
            return out

    def forward(
        self,
        signatures: list[tuple[str, ...]],
        values: torch.Tensor,
    ) -> torch.Tensor:
        return self.normalize(signatures, values)

    def get_extra_state(self) -> dict[str, Any]:
        stats_serialized = {}
        for k, (count, mean, M2) in self.stats.items():
            stats_serialized[k] = {
                "count": count,
                "mean": mean.cpu() if isinstance(mean, torch.Tensor) else mean,
                "M2": M2.cpu() if isinstance(M2, torch.Tensor) else M2,
            }
        return {"stats": stats_serialized}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        stats_serialized = state.get("stats", {})
        self.stats = {}
        for k, v in stats_serialized.items():
            mean = v["mean"]
            M2 = v["M2"]
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean)
            if not isinstance(M2, torch.Tensor):
                M2 = torch.tensor(M2)
            self.stats[k] = (int(v["count"]), mean, M2)
        self._invalidate_cache()
