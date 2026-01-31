"""
Per-signature нормализация наблюдений.
Хранит бегущие статистики (mean/std) для каждой signature
"""

from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn


class SignatureNormalizerModule(nn.Module):
    """
    Пер-сигнатурная нормализация с бегущими статистиками.
    
    Статистики хранятся в _stats_buffer как сериализуемый dict и регистрируются
    как persistent state через _extra_state API.
    """

    def __init__(self):
        super().__init__()
        # key -> (count:int, mean:Tensor, M2:Tensor)
        self.stats: Dict[str, Tuple[int, torch.Tensor, torch.Tensor]] = {}

    def _apply(self, fn):
        """Перемещает тензоры в stats при вызове .to()/.cpu()/.cuda()"""
        super()._apply(fn)
        for key, (count, mean, M2) in self.stats.items():
            self.stats[key] = (count, fn(mean), fn(M2))
        return self

    @staticmethod
    def _sig_key(sig) -> str:
        if isinstance(sig, (list, tuple)):
            return "|".join(sig)
        return str(sig)

    def update(self, signatures: List[Tuple[str, ...]], values: torch.Tensor):
        """
        Обновляет статистики для каждой signature.
        
        Args:
            signatures: длина n_obs
            values: [batch, n_obs, history_len]
        """
        b, n, h = values.shape
        assert n == len(signatures), "signatures and values length mismatch"
        vals = values.detach()
        for i, sig in enumerate(signatures):
            key = self._sig_key(sig)
            v = vals[:, i, -1]  # [batch]
            count_new = v.numel()
            mean_new = v.mean()
            var_new = v.var(unbiased=False)
            M2_new = var_new * count_new
            if key not in self.stats:
                self.stats[key] = (
                    count_new,
                    mean_new,
                    M2_new,
                )
            else:
                count, mean, M2 = self.stats[key]
                delta = mean_new - mean
                total = count + count_new
                mean = mean + delta * (count_new / total)
                M2 = M2 + M2_new + delta * delta * count * count_new / total
                self.stats[key] = (total, mean, M2)

    def normalize(self, signatures: List[Tuple[str, ...]], values: torch.Tensor) -> torch.Tensor:
        """
        Нормализует значения используя накопленные статистики.
        
        Args:
            signatures: длина n_obs
            values: [batch, n_obs, history_len]
        
        Returns:
            Нормализованные значения той же формы
        """
        b, n, h = values.shape
        assert n == len(signatures), "signatures and values length mismatch"
        device = values.device
        out = torch.empty_like(values)
        for i, sig in enumerate(signatures):
            key = self._sig_key(sig)
            if key not in self.stats:
                out[:, i, :] = values[:, i, :]
            else:
                count, mean, M2 = self.stats[key]
                if count < 2:
                    out[:, i, :] = values[:, i, :]
                else:
                    # Перемещаем статистики на device входных данных
                    mean = mean.to(device)
                    M2 = M2.to(device)
                    var = M2 / count
                    std = torch.sqrt(var + 1e-8)
                    out[:, i, :] = (values[:, i, :] - mean) / std
        return out

    def forward(self, signatures: List[Tuple[str, ...]], values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: обновляет статистики в training mode и нормализует значения.
        
        Args:
            signatures: длина n_obs
            values: [batch, n_obs, history_len]
        
        Returns:
            Нормализованные значения той же формы
        """
        if self.training:
            with torch.no_grad():
                self.update(signatures, values)
        return self.normalize(signatures, values)

    def get_extra_state(self) -> Dict[str, Any]:
        """
        Сериализует stats для сохранения в state_dict.
        PyTorch автоматически вызывает это при state_dict().
        """
        stats_serialized = {}
        for k, (count, mean, M2) in self.stats.items():
            # Конвертируем тензоры в CPU для безопасной сериализации
            stats_serialized[k] = {
                "count": count,
                "mean": mean.cpu() if isinstance(mean, torch.Tensor) else mean,
                "M2": M2.cpu() if isinstance(M2, torch.Tensor) else M2,
            }
        return {"stats": stats_serialized}

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        """
        Восстанавливает stats из state_dict.
        PyTorch автоматически вызывает это при load_state_dict().
        """
        stats_serialized = state.get("stats", {})
        self.stats = {}
        for k, v in stats_serialized.items():
            mean = v["mean"]
            M2 = v["M2"]
            # Восстанавливаем тензоры если нужно
            if not isinstance(mean, torch.Tensor):
                mean = torch.tensor(mean)
            if not isinstance(M2, torch.Tensor):
                M2 = torch.tensor(M2)
            self.stats[k] = (int(v["count"]), mean, M2)
