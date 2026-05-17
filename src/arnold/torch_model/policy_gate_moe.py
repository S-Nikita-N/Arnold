"""
Gate MoE Policy — Frozen experts + trainable categorical gate.

Архитектура (по аналогии с Kinesis PolicyMOE):
  - N pre-trained Arnold LatticePolicy экспертов, веса заморожены.
  - Gate: MLP(state) → Linear(N) → Softmax → Categorical распределение.
  - Value net: отдельный MLP (того же размера что gate) → Linear(1).

Что обучается: только gate и value_net. Эксперты — frozen forward.

Действия:
  - Сэмплируется expert_idx из Categorical(gate(state)).
  - Continuous action = expert_idx-й эксперт(state) (deterministic mean).
  - В среду идёт continuous action.
  - В memory сохраняется expert_idx (для log_prob при update).

PPO применяется к Categorical (over expert_idx), как в Kinesis.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from arnold.torch_model.mlp import MLP
from arnold.torch_model.policy_lattice import LatticePolicy
from arnold.torch_model.normalization import SignatureNormalizerModule
from arnold.profiler import Profiler


class GateMoEPolicy(nn.Module):
    """
    Frozen-experts MoE policy:
      - experts: List[LatticePolicy] — replicas of pre-trained Arnold lattice
      - gate: MLP → softmax over N experts
      - value_net: separate MLP critic

    Только single-expert среда (по аналогии с MoELatticePolicy).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        expert_checkpoints: List[str],
        gate_units: Tuple[int, ...] = (1024, 512, 256),
        gate_activation: str = "silu",
        device: str = "cpu",
    ):
        super().__init__()

        if not expert_checkpoints:
            raise ValueError("GateMoEPolicy: expert_checkpoints is empty")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_experts = len(expert_checkpoints)

        # ── Frozen experts ───────────────────────────────────────────
        self.experts = nn.ModuleList()
        for i, ckpt_path in enumerate(expert_checkpoints):
            expert = self._load_expert(ckpt_path, state_dim, action_dim)
            for p in expert.parameters():
                p.requires_grad_(False)
            expert.eval()
            self.experts.append(expert)

        # ── Trainable normalizer (для gate и value) ──────────────────
        self.obs_normalizer = SignatureNormalizerModule()

        # ── Gate ─────────────────────────────────────────────────────
        self.gate_net = MLP(state_dim, gate_units, gate_activation)
        self.gate_head = nn.Linear(gate_units[-1], self.num_experts)
        nn.init.orthogonal_(self.gate_head.weight, gain=0.1)
        nn.init.zeros_(self.gate_head.bias)

        # ── Value net ────────────────────────────────────────────────
        self.value_net = MLP(state_dim, gate_units, gate_activation)
        self.value_head = nn.Linear(gate_units[-1], 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.zero_()

        # ── Profiler ─────────────────────────────────────────────────
        self.profiler = Profiler()

    @staticmethod
    def _load_expert(
        checkpoint_path: str, state_dim: int, action_dim: int,
    ) -> LatticePolicy:
        """Load Arnold LatticePolicy from checkpoint. MLP shape inferred."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        policy_state = ckpt["policy"]

        layer_dims = []
        i = 0
        while f"net.affine_layers.{i}.weight" in policy_state:
            layer_dims.append(policy_state[f"net.affine_layers.{i}.weight"].shape[0])
            i += 1
        if not layer_dims:
            raise ValueError(
                f"Cannot infer MLP from {checkpoint_path}: "
                f"no 'net.affine_layers.*.weight' keys"
            )

        expert = LatticePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            mlp_units=tuple(layer_dims),
        )
        expert.load_state_dict(policy_state, strict=False)
        return expert

    def value_parameters(self):
        yield from self.value_net.parameters()
        yield from self.value_head.parameters()

    def policy_parameters(self):
        """Gate + normalizer (experts are frozen, value is separate)."""
        yield from self.gate_net.parameters()
        yield from self.gate_head.parameters()

    def set_profiler(self, profiler: Profiler) -> None:
        self.profiler = profiler

    # ------------------------------------------------------------------
    #  Forward — общий формат для trainer
    # ------------------------------------------------------------------

    def forward(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List,
        action_signatures: List,
        expert_name: str,
        return_std: bool = True,
        return_value: bool = True,
    ) -> Tuple:
        """
        Returns (gate_logits, None, None, value, None) — формат как у LatticePolicy
        но cov_factor/diag_std/latent_std не определены (Categorical).

        gate_logits размерности (B, num_experts).
        """
        p = self.profiler

        # ── Normalize ───────────────────────────────────────────────
        with p.section("normalizer"):
            obs_norm = self.obs_normalizer(obs_signatures, obs_timeseries)
            x = obs_norm[:, :, -1]

        # ── Gate ────────────────────────────────────────────────────
        with p.section("gate_net"):
            gate_h = self.gate_net(x)
            gate_logits = self.gate_head(gate_h)

        # ── Value ───────────────────────────────────────────────────
        value = None
        if return_value:
            with p.section("value_net"):
                value = self.value_head(self.value_net(x))

        # Возвращаем placeholder'ы для совместимости с интерфейсом
        return gate_logits, None, None, value, None

    # ------------------------------------------------------------------
    #  Distribution helpers
    # ------------------------------------------------------------------

    def build_action_dist(
        self,
        gate_logits: torch.Tensor,
        cov_factor: Optional[torch.Tensor] = None,
        diag_std: Optional[torch.Tensor] = None,
    ) -> Categorical:
        """Categorical over expert indices. cov_factor/diag_std игнорируются."""
        return Categorical(logits=gate_logits)

    def dist_log_prob(self, dist: Categorical, actions: torch.Tensor) -> torch.Tensor:
        """log_prob дискретного expert_idx. actions: (B,) или (B, 1) long."""
        idx = actions.squeeze(-1) if actions.dim() > 1 else actions
        return dist.log_prob(idx.long()).unsqueeze(-1)

    def dist_entropy(self, dist: Categorical) -> torch.Tensor:
        return dist.entropy().mean()

    # ------------------------------------------------------------------
    #  Action sampling
    # ------------------------------------------------------------------

    def get_action(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List,
        action_signatures: List,
        expert_name: str,
        deterministic: bool = False,
        return_std: bool = True,
        return_value: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        p = self.profiler

        gate_logits, _, _, value, _ = self.forward(
            obs_timeseries, obs_signatures, action_signatures,
            expert_name=expert_name,
            return_std=return_std,
            return_value=return_value,
        )

        with p.section("dist_build"):
            dist = self.build_action_dist(gate_logits)

        with p.section("dist_sample"):
            if deterministic:
                expert_idx = dist.probs.argmax(dim=-1)
            else:
                expert_idx = dist.sample()
            log_prob = dist.log_prob(expert_idx).unsqueeze(-1)

        # ── Forward через выбранных экспертов ──────────────────────
        with p.section("experts_forward"):
            env_action = self._run_selected_experts(
                obs_timeseries,
                obs_signatures,
                action_signatures,
                expert_name,
                expert_idx,
            )

        return env_action, expert_idx, log_prob, value

    def _run_selected_experts(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List,
        action_signatures: List,
        expert_name: str,
        expert_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward через каждого выбранного эксперта, собрать continuous mean.

        Подход: пройдёмся по уникальным индексам и батчируем по ним.
        """
        B = obs_timeseries.shape[0]
        env_action = torch.zeros(
            B, self.action_dim,
            device=obs_timeseries.device,
            dtype=obs_timeseries.dtype,
        )

        unique_idx = torch.unique(expert_idx)
        for idx_val in unique_idx.tolist():
            mask = expert_idx == idx_val
            if not mask.any():
                continue
            sub_obs = obs_timeseries[mask]
            expert = self.experts[idx_val]
            with torch.no_grad():
                mean, _, _, _, _ = expert.forward(
                    sub_obs, obs_signatures, action_signatures,
                    expert_name=expert_name,
                    return_std=False,
                    return_value=False,
                )
            env_action[mask] = mean.to(env_action.dtype)

        return env_action
