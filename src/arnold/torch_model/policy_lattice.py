"""
Lattice Policy — MLP с low-rank ковариацией (латентные факторы).

Архитектура:
  obs → normalizer → net(x)
                        ↓
        action_mean(latent) → mean [B, A]
                        ↓
        cov_factor = action_mean.weight * latent_std
                     [A, latent_dim] — expand → batch
                        ↓
        LowRankMultivariateNormal(mean, cov_factor, diag_std²)

Параметры:
  mlp_units      — слои основной MLP, последний = latent_dim
  mlp_activation — функция активации (default silu)
  fix_std        — заморозить log_std
  log_std_init   — начальное значение log_std
  min_diag_std   — минимальная диагональная σ

Только single-expert (multi-expert не поддерживается).
Интерфейс: forward → (mean, cov_factor, diag_std, value, latent_std).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LowRankMultivariateNormal
from typing import List, Tuple, Optional

from arnold.torch_model.dist_utils import safe_lrmvn_log_prob
from arnold.torch_model.mlp import MLP
from arnold.torch_model.normalization import SignatureNormalizerModule
from arnold.profiler import Profiler


# ---------------------------------------------------------------------------
#  LatticePolicy
# ---------------------------------------------------------------------------

class LatticePolicy(nn.Module):
    """
    Lattice Policy для Arnold. Только single-expert.

    Использует только текущий state (последний timestep),
    без истории. obs_timeseries [batch, n_obs, history_len] → [:, :, -1].

    Профилирование:
        self.profiler — Profiler (time + mem), настраивается через set_profiler()
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
        tanh_mean_squash: bool = False,
        tanh_mean_a: float = 1.2,
        tanh_mean_b: float = 0.1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = mlp_units[-1]
        self.min_diag_std = min_diag_std

        # Soft mean squashing: μ ← a·tanh(μ) + b·μ
        self.tanh_mean_squash = tanh_mean_squash
        self.tanh_mean_a = tanh_mean_a
        self.tanh_mean_b = tanh_mean_b

        # ── Normalizer ────────────────────────────────────────────────
        self.obs_normalizer = SignatureNormalizerModule()

        # ── Main MLP ─────────────────────────────────────────────────
        self.net = MLP(state_dim, mlp_units, mlp_activation)

        # ── Action head ──────────────────────────────────────────────
        self.action_mean = nn.Linear(self.latent_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.zero_()

        # ── Std parameters ───────────────────────────────────────────
        # [action_dim (diagonal) + latent_dim (low-rank cov)]
        self.log_std = nn.Parameter(
            torch.ones(1, action_dim + self.latent_dim) * log_std_init,
            requires_grad=not fix_std,
        )

        # ── Value network (отдельный) ────────────────────────────────
        self.value_net = MLP(state_dim, mlp_units, mlp_activation)
        self.value_head = nn.Linear(self.value_net.out_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.zero_()

        # ── Profiler ──────────────────────────────────────────────────
        self.profiler = Profiler()

    # ------------------------------------------------------------------
    #  Profiler management
    # ------------------------------------------------------------------

    def value_parameters(self):
        yield from self.value_net.parameters()
        yield from self.value_head.parameters()

    def policy_parameters(self):
        value_ids = {id(p) for p in self.value_parameters()}
        for p in self.parameters():
            if id(p) not in value_ids:
                yield p

    def set_profiler(self, profiler: Profiler) -> None:
        self.profiler = profiler

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List[Tuple[str, ...]],
        action_signatures: List[Tuple[str, ...]],
        expert_name: str,
        return_std: bool = True,
        return_value: bool = True,
    ) -> Tuple[
        torch.Tensor,           # mean       [batch, action_dim]
        Optional[torch.Tensor], # cov_factor [batch, action_dim, latent_dim]
        Optional[torch.Tensor], # diag_std   [batch, action_dim]
        Optional[torch.Tensor], # value      [batch, 1]
        Optional[torch.Tensor], # latent_std [latent_dim]
    ]:
        """
        Returns:
            mean, cov_factor, diag_std, value, latent_std
        """
        p = self.profiler

        # ── Normalizer ───────────────────────────────────────────────
        with p.section("normalizer"):
            obs_norm = self.obs_normalizer(obs_signatures, obs_timeseries)
            x = obs_norm[:, :, -1]

        # ── 1. Main MLP ─────────────────────────────────────────────
        with p.section("lattice_net"):
            latent = self.net(x)

        # ── 2. Action head ───────────────────────────────────────────
        with p.section("action_head"):
            action_mean = self.action_mean(latent)
            if self.tanh_mean_squash:
                self._last_raw_mean = action_mean.detach()
                action_mean = (
                    self.tanh_mean_a * torch.tanh(action_mean)
                    + self.tanh_mean_b * action_mean
                )
            else:
                self._last_raw_mean = None

        # ── 3. Covariance ────────────────────────────────────────────
        cov_factor = None
        diag_std = None
        latent_std = None
        if return_std:
            with p.section("cov_factor"):
                std = F.softplus(self.log_std) + self.min_diag_std
                diag_std = std[:, :self.action_dim]       # [1, A]
                latent_std = std[:, self.action_dim:].squeeze(0)  # [L]
                W = self.action_mean.weight               # [A, L]
                cov_factor = W * latent_std.unsqueeze(0)  # [A, L]

        # ── 4. Value network ─────────────────────────────────────────
        value = None
        if return_value:
            with p.section("value_net"):
                value = self.value_head(self.value_net(x))

        return action_mean, cov_factor, diag_std, value, latent_std

    # ------------------------------------------------------------------
    #  Action sampling
    # ------------------------------------------------------------------

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
        p = self.profiler

        mean, cov_factor, diag_std, value, latent_std = self.forward(
            obs_timeseries, obs_signatures, action_signatures,
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
            with p.section("dist_build"):
                dist = self.build_action_dist(mean, cov_factor, diag_std)
            with p.section("dist_sample"):
                action = dist.rsample()
                log_prob = safe_lrmvn_log_prob(dist, action).unsqueeze(-1)

        return action, log_prob, value

    # ------------------------------------------------------------------
    #  Distribution helpers
    # ------------------------------------------------------------------

    def build_action_dist(
        self,
        mean: torch.Tensor,
        cov_factor: torch.Tensor,
        diag_std: torch.Tensor,
    ) -> LowRankMultivariateNormal:
        p = self.profiler
        device = mean.device
        device_type = "cuda" if device.type == "cuda" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            with p.section("cast to float32"):
                mean_f = mean.float()
                cov_factor_f = cov_factor.float()
                diag_std_f = diag_std.float()
                cov_diag = diag_std_f.pow(2) + 1e-6

            with p.section("LowRankMultivariateNormal"):
                return LowRankMultivariateNormal(
                    loc=mean_f,
                    cov_factor=cov_factor_f,
                    cov_diag=cov_diag,
                )

    def compute_entropy(
        self,
        mean: torch.Tensor,
        cov_factor: torch.Tensor,
        diag_std: torch.Tensor,
    ) -> torch.Tensor:
        dist = self.build_action_dist(mean, cov_factor, diag_std)
        return dist.entropy().mean()

    def _compute_log_prob(
        self,
        actions: torch.Tensor,
        mean: torch.Tensor,
        cov_factor: torch.Tensor,
        diag_std: torch.Tensor,
    ) -> torch.Tensor:
        dist = self.build_action_dist(mean, cov_factor, diag_std)
        return safe_lrmvn_log_prob(dist, actions.float()).unsqueeze(-1)
