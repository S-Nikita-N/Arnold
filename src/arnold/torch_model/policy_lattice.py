"""
Lattice Policy — MLP с low-rank ковариацией (латентные факторы).

Адаптировано из Kinesis policy_lattice.py.
Ковариация: Σ = W·diag(σ²_latent)·W^T + diag(σ²_action) = F·F^T + diag(σ²_action).
Использует LowRankMultivariateNormal (как Transformer) — единый интерфейс.

Только single-expert (multi-expert не поддерживается).
Интерфейс: forward → (mean, cov_factor, diag_std, value).
"""

import torch
import torch.nn as nn
from torch.distributions import LowRankMultivariateNormal
from typing import List, Tuple, Optional

from arnold.torch_model.dist_utils import safe_lrmvn_log_prob

from arnold.torch_model.mlp import MLP
from arnold.torch_model.normalization import SignatureNormalizerModule


class LatticePolicy(nn.Module):
    """
    Lattice Policy для Arnold. Только single-expert.

    Как в Kinesis: использует только текущий state (последний timestep),
    без истории. obs_timeseries [batch, n_obs, history_len] → берём [:, :, -1].

    Интерфейс совместим с TransformerPolicy: (mean, cov_factor, diag_std, value).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        mlp_units: Tuple[int, ...] = (2048, 1536, 1024, 1024, 512, 512),
        mlp_activation: str = "silu",
        fix_std: bool = False,
        log_std_init: float = 0.0,
        min_diag_std: float = 1e-4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = mlp_units[-1]
        self.min_diag_std = min_diag_std

        self.obs_normalizer = SignatureNormalizerModule()

        self.net = MLP(state_dim, mlp_units, mlp_activation)

        self.action_mean = nn.Linear(self.latent_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.zero_()

        self.log_std = nn.Parameter(
            torch.ones(1, action_dim + self.latent_dim) * log_std_init,
            requires_grad=not fix_std,
        )

        self.value_net = MLP(state_dim, mlp_units, mlp_activation)
        self.value_head = nn.Linear(self.value_net.out_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.zero_()

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            mean: [batch, action_dim]
            cov_factor: [batch, action_dim, latent_dim]
            diag_std: [batch, action_dim] — σ = softplus(log_std) + min, уже готовый
            value: [batch, 1] или None
        """
        obs_norm = self.obs_normalizer(obs_signatures, obs_timeseries)
        x = obs_norm[:, :, -1]

        action_mean = self.action_mean(self.net(x))

        cov_factor = None
        diag_std = None
        latent_std = None
        if return_std:
            std = torch.nn.functional.softplus(self.log_std) + self.min_diag_std
            diag_std = std[:, : self.action_dim].expand(x.shape[0], -1)
            latent_std = std[:, self.action_dim:].squeeze(0)

            W = self.action_mean.weight
            cov_factor = (W * latent_std.unsqueeze(0)).unsqueeze(0).expand(
                x.shape[0], -1, -1
            )

        value = None
        if return_value:
            value = self.value_head(self.value_net(x))

        return action_mean, cov_factor, diag_std, value, latent_std

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
        mean, cov_factor, diag_std, value, latent_std = self.forward(
            obs_timeseries,
            obs_signatures,
            action_signatures,
            expert_name=expert_name,
            return_std=return_std,
            return_value=return_value,
        )

        if deterministic:
            action = mean
            log_prob = None if return_std else torch.zeros(mean.shape[0], 1, device=mean.device)
            
        else:
            if cov_factor is None:
                raise ValueError("return_std=False допустим только при deterministic=True")

            dist = self._build_action_dist(mean, cov_factor, diag_std)
            action = dist.rsample()
            log_prob = safe_lrmvn_log_prob(dist, action).unsqueeze(-1)

        return action, log_prob, value

    def _build_action_dist(
        self,
        mean: torch.Tensor,
        cov_factor: torch.Tensor,
        diag_std: torch.Tensor,
    ) -> LowRankMultivariateNormal:
        """diag_std уже готовый: σ = softplus(log_std) + min."""
        device = mean.device
        device_type = "cuda" if device.type == "cuda" else "cpu"
        autocast = torch.autocast(device_type=device_type, enabled=False)

        with autocast:
            mean_f = mean.float()
            cov_factor_f = cov_factor.float()
            diag_std_f = diag_std.float()
            cov_diag = diag_std_f.pow(2) + 1e-6

            return LowRankMultivariateNormal(
                loc=mean_f,
                cov_factor=cov_factor_f,
                cov_diag=cov_diag,
            )

    def _compute_entropy(
        self,
        mean: torch.Tensor,
        cov_factor: torch.Tensor,
        diag_std: torch.Tensor,
    ) -> torch.Tensor:
        dist = self._build_action_dist(mean, cov_factor, diag_std)
        return dist.entropy().mean()

    def _compute_log_prob(
        self,
        actions: torch.Tensor,
        mean: torch.Tensor,
        cov_factor: torch.Tensor,
        diag_std: torch.Tensor,
    ) -> torch.Tensor:
        dist = self._build_action_dist(mean, cov_factor, diag_std)
        return safe_lrmvn_log_prob(dist, actions.float()).unsqueeze(-1)
