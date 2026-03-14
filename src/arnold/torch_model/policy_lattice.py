"""
Lattice Policy — MLP с полной ковариационной матрицей (латентные факторы).

Адаптировано из Kinesis policy_lattice.py.
Ковариация: Σ = W·diag(σ²_latent)·W^T + diag(σ²_action).
Позволяет моделировать корреляции между действиями через латентное пространство.

Только single-expert (multi-expert не поддерживается).
Интерфейс совместим с Arnold Trainer (forward, get_action, _compute_log_prob).
"""

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from typing import List, Tuple, Optional

from arnold.torch_model.mlp import MLP
from arnold.torch_model.normalization import SignatureNormalizerModule


class LatticePolicy(nn.Module):
    """
    Lattice Policy для Arnold. Только single-expert.

    Как в Kinesis: использует только текущий state (последний timestep),
    без истории. obs_timeseries [batch, n_obs, history_len] → берём [:, :, -1].

    Вход: obs_timeseries [batch, n_obs, history_len]
    Выход: action_mean, log_std (диагональ для entropy), value.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        mlp_units: Tuple[int, ...] = (2048, 1536, 1024, 1024, 512, 512),
        mlp_activation: str = "silu",
        fix_std: bool = False,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # latent_dim = размер последнего слоя MLP (для ковариации W @ diag(σ²) @ W^T)
        self.latent_dim = mlp_units[-1]

        self.obs_normalizer = SignatureNormalizerModule()

        self.net = MLP(state_dim, mlp_units, mlp_activation)

        self.action_mean = nn.Linear(self.latent_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.zero_()

        self.log_std = nn.Parameter(
            torch.ones(1, action_dim + self.latent_dim) * log_std_init,
            requires_grad=not fix_std,
        )

        # Value head: отдельный MLP на том же входе (как в Kinesis)
        self.value_net = MLP(state_dim, mlp_units, mlp_activation)
        self.value_head = nn.Linear(self.value_net.out_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.zero_()

        self._last_dist: Optional[MultivariateNormal] = None
        self.profiler = None

    def enable_profiling(self, profiler) -> None:
        self.profiler = profiler

    def disable_profiling(self) -> None:
        self.profiler = None

    def forward(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List[Tuple[str, ...]],
        action_signatures: List[Tuple[str, ...]],
        expert_name: str,
        return_std: bool = True,
        return_value: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            obs_timeseries: [batch, n_obs, history_len]
            obs_signatures, action_signatures: для нормализатора (action не используется)
            expert_name: для совместимости с интерфейсом

        Returns:
            action_mean: [batch, action_dim]
            log_std: [batch, action_dim] — диагональ ковариации для entropy
            value: [batch, 1] или None
        """
        # 1. Нормализация
        obs_norm = self.obs_normalizer(obs_signatures, obs_timeseries)
        # Kinesis: только последний timestep (текущий state), без истории
        x = obs_norm[:, :, -1]  # [batch, n_obs] = [batch, state_dim]

        # 2. Policy backbone (h: [batch, latent_dim])
        action_mean = self.action_mean(self.net(x))

        # 3. Lattice covariance: Σ = W @ diag(σ²_latent) @ W^T + diag(σ²_action)
        std = torch.exp(self.log_std)
        action_var = std[:, :self.action_dim].pow(2)  # [1, action_dim]
        latent_var = std[:, self.action_dim:].pow(2)  # [1, latent_dim]

        W = self.action_mean.weight  # [action_dim, latent_dim]
        # (W * latent_var) @ W^T -> [action_dim, action_dim], then expand to batch
        sigma_mat = (W * latent_var).matmul(W.T)
        sigma_mat = sigma_mat.unsqueeze(0).expand(x.shape[0], -1, -1).clone()
        sigma_mat[:, torch.arange(self.action_dim), torch.arange(self.action_dim)] += (
            action_var.squeeze(0)
        )
        sigma_mat = sigma_mat + 1e-6 * torch.eye(
            self.action_dim, device=sigma_mat.device, dtype=sigma_mat.dtype
        )

        # log_std для entropy (диагональ ковариации), [batch, action_dim]
        sigma_diag = action_var.squeeze(0) + (W * latent_var).matmul(W.T).diag()
        log_std_diag = (0.5 * torch.log(sigma_diag + 1e-8)).unsqueeze(0).expand(
            x.shape[0], -1
        )

        dist = MultivariateNormal(action_mean, covariance_matrix=sigma_mat)
        self._last_dist = dist

        value = None
        if return_value:
            value = self.value_head(self.value_net(x))

        log_std_out = log_std_diag if return_std else None
        return action_mean, log_std_out, value

    def get_action(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List[Tuple[str, ...]],
        action_signatures: List[Tuple[str, ...]],
        expert_name: str,
        deterministic: bool = False,
        return_std: bool = True,
        return_value: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        mean, log_std, value = self.forward(
            obs_timeseries,
            obs_signatures,
            action_signatures,
            expert_name=expert_name,
            return_std=return_std,
            return_value=return_value,
        )

        if deterministic:
            action = mean
            log_prob = torch.zeros(mean.shape[0], 1, device=mean.device)
        else:
            action = self._last_dist.rsample()
            log_prob = self._last_dist.log_prob(action).unsqueeze(-1)

        return action, log_prob, value

    def _compute_log_prob(
        self,
        actions: torch.Tensor,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Log probability действий. Для Lattice используем _last_dist.
        """
        if self._last_dist is not None:
            return self._last_dist.log_prob(actions).unsqueeze(-1)
        # Fallback: diagonal Gaussian (если _last_dist не установлен)
        var = (log_std * 2).exp()
        log_prob = -0.5 * (
            torch.log(torch.tensor(2 * 3.14159265359, device=actions.device, dtype=actions.dtype))
            + 2 * log_std
            + (actions - mean).pow(2) / var
        )
        return log_prob.sum(dim=-1, keepdim=True)
