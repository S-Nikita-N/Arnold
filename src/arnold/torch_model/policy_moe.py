"""
MoE-Lattice Policy — Shared Expert + N Routed Experts, Low-Rank Covariance.

Архитектура:
  obs → normalizer → shared_trunk(x) → h   (h = x если shared_trunk пустой)
                         ↓
        ┌────────────────┴────────────────────────────────────────────────────┐
        │  Shared Expert         │  Routing (top-k из N)  │  Gate (отдельная) │
        │  net(h) → e_shared     │  expert_i(h) → [B, A]  │  gate_net(h)      │
        │  [B, latent_dim]       │  expert_heads → [B, A]  │  → [B, num_exp]   │
        │  action_head(e_shared) │  head.weight → W_i      │                   │
        │  → shared_mean [B, A]  │                        │                   │
        └────────────────┬────────────────────────────────────────────────────┘
                         ↓
          mean       = α·shared_mean + (1-α)·routed_mean
          cov_factor = α·W_shared   + (1-α)·Σ(w_i·W_i)   (всё * latent_std)
                         ↓
          LowRankMultivariateNormal(mean, cov_factor, diag_std²)

  Требуется: shared_expert_units[-1] == expert_units[-1] (общий latent_dim).

Параметры:
  shared_units        — слои shared_trunk (общий для всех путей), по умолчанию (2048,1536,1024)
  shared_expert_units — слои shared expert body, последний = latent_dim (default (256,64))
  num_experts         — размер пула routing-экспертов (default 5)
  top_k               — сколько routing-экспертов активировать (default 1 → 1/5 пула)
  expert_units        — слои каждого routing-эксперта, последний = latent_dim (default (512,512))
  gate_units          — слои отдельной gate MLP (input = h), default (512, 256)
  value_units         — слои отдельной value MLP, default (2048,1536,1024,512,512)
  alpha               — вес shared expert (default 0.3), routing = 1-alpha
  load_balance_weight — вес Switch Transformer LB loss (default 0.01)
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
#  MoELatticePolicy
# ---------------------------------------------------------------------------

class MoELatticePolicy(nn.Module):
    """
    Shared Expert + N Routed Experts MoE Policy с low-rank ковариацией.

    Ковариация строится из ВСЕХ экспертов (shared + routed), взвешивается gate:
        cov_factor = α·W_shared + (1-α)·Σ(w_i·W_i), всё умножено на latent_std
        shape:       [batch, action_dim, latent_dim]  — per-sample через gate weights

    Требование: shared_expert_units[-1] == expert_units[-1] (единый latent_dim).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        shared_units: Tuple[int, ...],
        shared_expert_units: Tuple[int, ...],
        num_experts: int,
        top_k: int,
        expert_units: Tuple[int, ...],
        gate_units: Tuple[int, ...],
        value_units: Tuple[int, ...],
        alpha: float = 0.3,
        load_balance_weight: float = 0.01,
        fix_std: bool = False,
        log_std_init: float = 0.0,
        min_diag_std: float = 1e-4,
        activation: str = "silu",
        grad_checkpoint_cov: bool = False,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha
        self.load_balance_weight = load_balance_weight
        self.min_diag_std = min_diag_std
        self.grad_checkpoint_cov = grad_checkpoint_cov

        # ── Normalizer ────────────────────────────────────────────────
        self.obs_normalizer = SignatureNormalizerModule()

        # ── Shared trunk ─────────────────────────────────────────────
        if shared_units:
            self.shared_trunk = MLP(state_dim, shared_units, activation)
            trunk_out_dim = shared_units[-1]
        else:
            self.shared_trunk = None
            trunk_out_dim = state_dim

        # ── Shared expert ─────────────────────────────────────────────
        if not shared_expert_units:
            raise ValueError("shared_expert_units не может быть пустым")
        if not expert_units:
            raise ValueError("expert_units не может быть пустым")
        if shared_expert_units[-1] != expert_units[-1]:
            raise ValueError(
                f"shared_expert_units[-1] ({shared_expert_units[-1]}) != "
                f"expert_units[-1] ({expert_units[-1]}): "
                f"latent_dim должен совпадать для cov_factor aggregation"
            )

        self.latent_dim = shared_expert_units[-1]

        self.shared_expert_net = MLP(trunk_out_dim, shared_expert_units, activation)
        self.shared_expert_head = nn.Linear(self.latent_dim, action_dim)
        self.shared_expert_head.weight.data.mul_(0.1)
        self.shared_expert_head.bias.data.zero_()

        # ── Gate network ──────────────────────────────────────────────
        if not gate_units:
            raise ValueError("gate_units не может быть пустым")

        self.gate = nn.Sequential(
            MLP(trunk_out_dim, gate_units, activation),
            nn.Linear(gate_units[-1], num_experts),
        )

        # ── Routing experts ───────────────────────────────────────────
        self.expert_nets = nn.ModuleList()
        self.expert_heads = nn.ModuleList()
        for _ in range(num_experts):
            net = MLP(trunk_out_dim, expert_units, activation)
            head = nn.Linear(self.latent_dim, action_dim)
            head.weight.data.mul_(0.1)
            head.bias.data.zero_()
            self.expert_nets.append(net)
            self.expert_heads.append(head)

        # ── Std parameters ────────────────────────────────────────────
        # [action_dim (diagonal) + latent_dim (low-rank cov)]
        self.log_std = nn.Parameter(
            torch.ones(1, action_dim + self.latent_dim) * log_std_init,
            requires_grad=not fix_std,
        )

        # ── Value network (отдельная MLP) ─────────────────────────────
        if not value_units:
            raise ValueError("value_units не может быть пустым")

        self.value_net = MLP(state_dim, value_units, activation)
        self.value_head = nn.Linear(value_units[-1], 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.zero_()

        # ── Profiler ──────────────────────────────────────────────────
        self.profiler = Profiler()

    # ------------------------------------------------------------------
    #  Profiler management
    # ------------------------------------------------------------------

    def set_profiler(self, profiler: Profiler) -> None:
        self.profiler = profiler

    # ------------------------------------------------------------------
    #  Gate
    # ------------------------------------------------------------------

    def gate_forward(self, h: torch.Tensor):
        logits = self.gate(h)
        gate_probs = F.softmax(logits, dim=-1)
        tie_breaker = torch.rand_like(gate_probs) * 1e-7
        _, top_k_indices = torch.topk(gate_probs + tie_breaker, self.top_k, dim=-1)
        top_k_vals = gate_probs.gather(1, top_k_indices)
        top_k_indices = top_k_indices.long()
        top_k_weights = top_k_vals / (top_k_vals.sum(dim=-1, keepdim=True) + 1e-8)
        return top_k_weights, top_k_indices, gate_probs

    def compute_load_balance_loss(
        self,
        top_k_indices: torch.Tensor,
        gate_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Switch Transformer LB loss."""
        N = self.num_experts
        flat = top_k_indices.view(-1)
        f = torch.bincount(flat, minlength=N).float() / flat.numel()
        P = gate_probs.mean(dim=0)
        return (N * (f * P).sum()).unsqueeze(0)

    # ------------------------------------------------------------------
    #  Covariance factor (выделено для gradient checkpointing)
    # ------------------------------------------------------------------

    def _compute_cov_factor(
        self,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fused cov_factor computation в fp32.

        Возвращает (cov_factor, diag_std, latent_std):
            cov_factor  [B, A, L]
            diag_std    [B, A]
            latent_std  [L]

        Трактует shared как "expert 0" с фиксированным весом α,
        всё считается одним matmul [B, E+1] @ [E+1, A*L].

        Выделено в отдельный метод ради torch.utils.checkpoint.checkpoint:
        при grad_checkpoint_cov=True forward activations не хранятся,
        backward вызывает этот метод заново.
        """
        batch_size = top_k_indices.shape[0]
        device = self.log_std.device
        device_type = "cuda" if device.type == "cuda" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):
            std = F.softplus(self.log_std.float()) + self.min_diag_std
            diag_std = std[:, :self.action_dim].expand(batch_size, -1)
            latent_std = std[:, self.action_dim:].squeeze(0)

            W_shared = self.shared_expert_head.weight

            mix = torch.zeros(
                batch_size, self.num_experts + 1,
                dtype=torch.float32, device=device,
            )
            mix[:, 0] = self.alpha
            mix[:, 1:].scatter_(
                1, top_k_indices,
                ((1.0 - self.alpha) * top_k_weights).float(),
            )

            W_all = torch.cat(
                [
                    W_shared.unsqueeze(0),
                    torch.stack([eh.weight for eh in self.expert_heads]),
                ],
                dim=0,
            )
            W_all_flat = (W_all * latent_std[None, None, :]).view(
                self.num_experts + 1, -1,
            )

            cov_factor = (mix @ W_all_flat).view(
                batch_size, self.action_dim, self.latent_dim,
            )

        return cov_factor, diag_std, latent_std

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
        torch.Tensor,           # mean          [batch, action_dim]
        Optional[torch.Tensor], # cov_factor    [batch, action_dim, latent_dim]
        Optional[torch.Tensor], # diag_std      [batch, action_dim]
        Optional[torch.Tensor], # value         [batch, 1]
        Optional[torch.Tensor], # latent_std    [latent_dim]
        torch.Tensor,           # load_balance_loss [1]
    ]:
        """
        Returns:
            mean, cov_factor, diag_std, value, latent_std, load_balance_loss
        """
        p = self.profiler

        # ── Normalizer ───────────────────────────────────────────────
        with p.section("normalizer"):
            obs_norm = self.obs_normalizer(obs_signatures, obs_timeseries)
            x = obs_norm[:, :, -1]    # [batch, state_dim] — последний timestep
            batch_size = x.shape[0]

        if x.shape[1] != self.state_dim:
            raise RuntimeError(
                f"MoE state_dim mismatch: policy expects {self.state_dim}, "
                f"got obs with {x.shape[1]} elements "
                f"(obs_timeseries.shape={obs_timeseries.shape}, "
                f"n_signatures={len(obs_signatures)})"
            )

        # ── 1. Shared trunk ──────────────────────────────────────────
        with p.section("shared_trunk"):
            h = self.shared_trunk(x) if self.shared_trunk is not None else x

        # ── 2. Shared expert ─────────────────────────────────────────
        with p.section("shared_expert"):
            e_shared = self.shared_expert_net(h)            # [batch, latent_dim]
            shared_mean = self.shared_expert_head(e_shared) # [batch, action_dim]

        # ── 3. Gate routing (отдельная MLP на h) ─────────────────────
        with p.section("gate_forward"):
            top_k_weights, top_k_indices, gate_probs = self.gate_forward(h)
            load_balance_loss = self.compute_load_balance_loss(top_k_indices, gate_probs)

        # ── 4. Routing experts (sparse) ──────────────────────────────
        flat_indices = top_k_indices.view(-1)
        flat_h = h.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, h.size(-1))
        flat_route_out = torch.zeros(
            batch_size * self.top_k, self.action_dim,
            dtype=h.dtype, device=h.device,
        )

        with p.section("routing_experts"):
            for i in range(self.num_experts):
                idx = (flat_indices == i).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                e_out = self.expert_nets[i](flat_h[idx])
                head_out = self.expert_heads[i](e_out).to(flat_route_out.dtype)
                flat_route_out.index_add_(0, idx, head_out)

        # ── 5. Combine shared + routed → mean ────────────────────────
        with p.section("mean_combine"):
            route_means = flat_route_out.view(batch_size, self.top_k, self.action_dim)
            routed_mean = (route_means * top_k_weights.unsqueeze(-1)).sum(dim=1)
            mean = self.alpha * shared_mean + (1.0 - self.alpha) * routed_mean

        # ── 6. Covariance from ALL experts (shared + routed) ──────────
        cov_factor = None
        diag_std = None
        latent_std = None
        if return_std:
            with p.section("cov_factor"):
                # При grad_checkpoint_cov: не храним forward activations этого
                # блока — пересчитываем в backward. Экономит ~5GB на L=256
                # ценой +30% compute на backward. В no_grad контекстах
                # (sampling/eval) checkpoint сам делает прямой вызов.
                if self.grad_checkpoint_cov:
                    cov_factor, diag_std, latent_std = torch.utils.checkpoint.checkpoint(
                        self._compute_cov_factor,
                        top_k_indices, top_k_weights,
                        use_reentrant=False,
                    )
                else:
                    cov_factor, diag_std, latent_std = self._compute_cov_factor(
                        top_k_indices, top_k_weights,
                    )

        # Кеш норм для диагностики (detach, без влияния на граф)
        with torch.no_grad():
            s_norm = shared_mean.detach().norm(dim=-1).mean().item()
            r_norm = routed_mean.detach().norm(dim=-1).mean().item()
            s_contrib = self.alpha * s_norm
            r_contrib = (1.0 - self.alpha) * r_norm
            total = s_contrib + r_contrib + 1e-8
            self._expert_stats = {
                "shared_mean_norm": s_norm,
                "routed_mean_norm": r_norm,
                "shared_mean_frac": s_contrib / total,
            }
            if cov_factor is not None:
                # Norm approximation: avoid materializing [B, A, L] routed_W
                # by using per-expert weight norms + mean gate usage.
                # E[||routed_W[b]||] ≈ Σ_i mean_probs[i] · ||W_i||
                s_cov_norm = self.shared_expert_head.weight.detach().norm().item()
                W_exp_norms = torch.stack(
                    [eh.weight.detach().norm() for eh in self.expert_heads]
                )
                mean_probs = gate_probs.detach().mean(dim=0)
                r_cov_norm = (mean_probs * W_exp_norms).sum().item()
                sc = self.alpha * s_cov_norm
                rc = (1.0 - self.alpha) * r_cov_norm
                ct = sc + rc + 1e-8
                self._expert_stats["shared_cov_norm"] = s_cov_norm
                self._expert_stats["routed_cov_norm"] = r_cov_norm
                self._expert_stats["shared_cov_frac"] = sc / ct

        # ── 7. Value network ─────────────────────────────────────────
        value = None
        if return_value:
            with p.section("value_net"):
                value = self.value_head(self.value_net(x))

        return mean, cov_factor, diag_std, value, latent_std, load_balance_loss

    # ------------------------------------------------------------------
    #  Gate diagnostics
    # ------------------------------------------------------------------

    def get_gate_stats(self, x: torch.Tensor) -> dict:
        """Per-expert routing fractions, mean probabilities, gate entropy."""
        h = self.shared_trunk(x) if self.shared_trunk is not None else x
        top_k_weights, top_k_indices, gate_probs = self.gate_forward(h)
        N = self.num_experts
        one_hot = F.one_hot(top_k_indices, N).float()
        routing_fracs = one_hot.sum(dim=1).mean(dim=0)
        mean_probs = gate_probs.mean(dim=0)
        gate_entropy = -(gate_probs * (gate_probs + 1e-8).log()).sum(dim=-1).mean()
        balance_entropy = -(mean_probs * (mean_probs + 1e-8).log()).sum()
        return {
            "routing_fracs": routing_fracs.detach(),
            "mean_probs": mean_probs.detach(),
            "gate_entropy": gate_entropy.item(),
            "balance_entropy": balance_entropy.item(),
        }

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
