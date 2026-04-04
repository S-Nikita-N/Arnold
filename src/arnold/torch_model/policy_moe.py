"""
MoE-Lattice Policy — Mixture of Experts с low-rank ковариацией.

Архитектура:
  obs → normalizer → shared_trunk (configurable MLP layers)
                          ↓
        gate(h) → top-k softmax weights
        expert_1(h) ──┐
        expert_2(h) ──┼── weighted sum → action_mean
        expert_N(h) ──┘
                          ↓
        Lattice(action_mean, W_lattice, log_std) → LowRankMultivariateNormal

Soft routing (top-k из N экспертов). Load balancing loss для предотвращения коллапса.
Lattice weight aggregation: weighted (взвешенная сумма W) или max_score (W от top-1 эксперта).

Single-expert only (как LatticePolicy). Multi-expert через Arnold trainer не поддерживается.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LowRankMultivariateNormal
from typing import List, Tuple, Optional

from arnold.torch_model.dist_utils import safe_lrmvn_log_prob

from arnold.torch_model.mlp import MLP
from arnold.torch_model.normalization import SignatureNormalizerModule


class MoELatticePolicy(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_experts: int = 5,
        top_k: int = 2,
        shared_units: Tuple[int, ...] = (2048, 1536, 1024),
        expert_units: Tuple[int, ...] = (512, 512),
        gate_units: Tuple[int, ...] = (512, 256),
        activation: str = "silu",
        lattice_mode: str = "weighted",  # "weighted" | "max_score"
        load_balance_weight: float = 0.01,
        fix_std: bool = False,
        log_std_init: float = 0.0,
        min_diag_std: float = 1e-4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.lattice_mode = lattice_mode
        self.load_balance_weight = load_balance_weight
        self.min_diag_std = min_diag_std

        self.obs_normalizer = SignatureNormalizerModule()

        # Shared trunk (может быть пустым если shared_units = ())
        if shared_units:
            self.shared_trunk = MLP(state_dim, shared_units, activation)
            trunk_out_dim = shared_units[-1]
        else:
            self.shared_trunk = None
            trunk_out_dim = state_dim

        # Expert networks: каждый expert = MLP → Linear(action_dim)
        expert_out_dim = expert_units[-1] if expert_units else trunk_out_dim
        self.latent_dim = expert_out_dim

        self.expert_nets = nn.ModuleList()
        self.expert_heads = nn.ModuleList()
        for _ in range(num_experts):
            if expert_units:
                net = MLP(trunk_out_dim, expert_units, activation)
            else:
                net = nn.Identity()
            head = nn.Linear(expert_out_dim, action_dim)
            head.weight.data.mul_(0.1)
            head.bias.data.zero_()
            self.expert_nets.append(net)
            self.expert_heads.append(head)

        # Gate network
        self.gate = nn.Sequential(
            MLP(trunk_out_dim, gate_units, activation),
            nn.Linear(gate_units[-1], num_experts),
        )

        # Lattice: log_std = [action_dim (diagonal) + latent_dim (low-rank)]
        self.log_std = nn.Parameter(
            torch.ones(1, action_dim + self.latent_dim) * log_std_init,
            requires_grad=not fix_std,
        )

        # Value network (отдельный от MoE)
        value_units = shared_units + expert_units if shared_units else expert_units
        self.value_net = MLP(state_dim, value_units, activation)
        self.value_head = nn.Linear(value_units[-1], 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.zero_()

        self.profiler = None

    def enable_profiling(self, profiler) -> None:
        self.profiler = profiler

    def disable_profiling(self) -> None:
        self.profiler = None

    def gate_forward(self, h: torch.Tensor):
        """
        Gate → top-k routing.

        Returns:
            top_k_weights: [batch, top_k] — normalized weights for selected experts
            top_k_indices: [batch, top_k] — indices of selected experts
            gate_probs: [batch, num_experts] — full softmax (for load balancing)
        """
        logits = self.gate(h)
        gate_probs = F.softmax(logits, dim=-1)

        top_k_vals, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # Renormalize top-k weights to sum to 1
        top_k_weights = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + 1e-8)

        return top_k_weights, top_k_indices, gate_probs

    def compute_load_balance_loss(
        self,
        top_k_indices: torch.Tensor,
        gate_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LB loss from cached gate state )."""
        N = self.num_experts
        # f_i: fraction of tokens where expert i is in top-k
        one_hot = F.one_hot(top_k_indices, N).float()  # [batch, top_k, N]
        f = one_hot.sum(dim=1).mean(dim=0)  # [N]
        # P_i: mean gate probability
        P = gate_probs.mean(dim=0)  # [N]
        return (N * (f * P).sum()).unsqueeze(0)

    def forward(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List[Tuple[str, ...]],
        action_signatures: List[Tuple[str, ...]],
        expert_name: str,
        return_std: bool = True,
        return_value: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
            mean, cov_factor, diag_std, value, latent_std — как у других policy
            load_balance_loss: [1] — Switch Transformer LB loss (MoE-specific)
        """
        obs_norm = self.obs_normalizer(obs_signatures, obs_timeseries)
        x = obs_norm[:, :, -1]  # [batch, state_dim] — последний timestep
        batch_size = x.shape[0]

        if x.shape[1] != self.state_dim:
            raise RuntimeError(
                f"MoE state_dim mismatch: policy expects {self.state_dim}, "
                f"got obs with {x.shape[1]} elements "
                f"(obs_timeseries shape={obs_timeseries.shape}, "
                f"n_signatures={len(obs_signatures)})"
            )

        # Shared trunk
        h = self.shared_trunk(x) if self.shared_trunk is not None else x

        # Gate routing
        top_k_weights, top_k_indices, gate_probs = self.gate_forward(h)
        load_balance_loss = self.compute_load_balance_loss(gate_probs, top_k_indices)

        # Sparse expert computation: each expert processes only its routed samples
        selected_means = h.new_zeros(batch_size, self.top_k, self.action_dim)
        for k in range(self.top_k):
            slot_indices = top_k_indices[:, k]  # [batch] — expert index for this slot
            for i in range(self.num_experts):
                mask = slot_indices == i
                if not mask.any():
                    continue
                e_out = self.expert_nets[i](h[mask])
                selected_means[mask, k] = self.expert_heads[i](e_out)

        # Weighted sum
        w = top_k_weights.unsqueeze(-1)  # [batch, top_k, 1]
        mean = (selected_means * w).sum(dim=1)  # [batch, action_dim]

        cov_factor = None
        diag_std = None
        latent_std = None
        if return_std:
            std = F.softplus(self.log_std) + self.min_diag_std
            diag_std = std[:, :self.action_dim].expand(batch_size, -1)
            latent_std = std[:, self.action_dim:].squeeze(0)  # [latent_dim]

            # Stack all expert W matrices: [num_experts, action_dim, latent_dim]
            all_W = torch.stack([head.weight for head in self.expert_heads])

            if self.lattice_mode == "weighted":
                # Gather top-k W matrices and do weighted sum
                # top_k_indices: [batch, top_k]
                idx_W = top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(
                    -1, -1, self.action_dim, self.latent_dim,
                )  # [batch, top_k, action_dim, latent_dim]
                all_W_expanded = all_W.unsqueeze(0).expand(batch_size, -1, -1, -1)
                selected_W = torch.gather(all_W_expanded, 1, idx_W)  # [batch, top_k, action_dim, latent_dim]
                # Weighted sum: [batch, top_k, 1, 1] * [batch, top_k, action_dim, latent_dim]
                W_combined = (selected_W * top_k_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            else:
                # max_score: W от top-1 эксперта
                top1_idx = top_k_indices[:, 0]  # [batch]
                W_combined = all_W[top1_idx]  # [batch, action_dim, latent_dim]

            cov_factor = W_combined * latent_std.unsqueeze(0).unsqueeze(0)

        value = None
        if return_value:
            value = self.value_head(self.value_net(x))

        return mean, cov_factor, diag_std, value, latent_std, load_balance_loss

    def get_gate_stats(self, h: torch.Tensor) -> dict:
        """Gate diagnostics: per-expert routing fractions and probabilities."""
        top_k_weights, top_k_indices, gate_probs = self.gate_forward(h)
        N = self.num_experts
        one_hot = F.one_hot(top_k_indices, N).float()
        routing_fracs = one_hot.sum(dim=1).mean(dim=0)  # [N]
        mean_probs = gate_probs.mean(dim=0)  # [N]
        gate_entropy = -(gate_probs * (gate_probs + 1e-8).log()).sum(dim=-1).mean()
        return {
            "routing_fracs": routing_fracs.detach(),
            "mean_probs": mean_probs.detach(),
            "gate_entropy": gate_entropy.item(),
        }

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
        mean, cov_factor, diag_std, value, _, _lb = self.forward(
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
