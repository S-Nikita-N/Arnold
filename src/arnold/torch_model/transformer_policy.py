"""
Arnold Transformer Policy.

Encoder-Decoder Transformer для управления мышцами.
- Body Tokenizer группирует наблюдения по телам (~70 токенов вместо ~953)
- Shared Encoder обрабатывает body-level embeddings
- Action Decoder генерирует muscle activations (с опциональной грануляцией)
- Value Decoder генерирует state value

Action granulation (action_granulation != "none"):
- Вместо A muscle queries используются G group queries (G << A)
- ActionTokenizer раскрывает group embeddings в per-muscle activations
- Сложность decoder: O(G²) вместо O(A²)

Action distribution: LowRankMultivariateNormal, Lattice-like.
- cov_mode "action_out": W = action_out, cov_factor = action_out * σ_latent
- cov_mode "head": W = action_factor_head(action_out), cov_factor = W * σ_latent
- log_std [action_dim + latent_dim] как в Lattice MLP

Multi-expert: поддерживает forward с expert_name для корректного
выбора index-буферов в BodyTokenizer, ActionTokenizer и group signatures.
"""

import torch
import torch.nn as nn
from torch.distributions import LowRankMultivariateNormal
from typing import List, Dict, Optional, Tuple, Union
import logging

from arnold.torch_model.sensorimotor_vocabulary import SensorimotorVocabulary
from arnold.torch_model.normalization import SignatureNormalizerModule
from arnold.torch_model.body_tokenizer import BodyTokenizer
from arnold.torch_model.action_tokenizer import ActionTokenizer
from arnold.action_parser import MuscleGrouping
from arnold.observation_parser import BodyGroup
from arnold.profiler import SamplingProfiler

logger = logging.getLogger(__name__)


class TransformerPolicy(nn.Module):
    """
    Arnold Transformer Policy с body-level tokenization.

    Вместо 953 скалярных токенов (один на элемент наблюдения)
    используется ~70 body-level токенов. Self-attention: 70² vs 953² = 185x быстрее.
    """

    def __init__(
        self,
        vocab: SensorimotorVocabulary,
        groups: Union[List[BodyGroup], Dict[str, List[BodyGroup]]],
        history_len: int = 5,
        embed_dim: int = 128,
        ff_dim: int = 512,
        num_heads: int = 4,
        num_enc_layers: int = 6,
        num_act_dec_layers: int = 6,
        num_val_dec_layers: int = 1,
        dropout: float = 0.0,
        detached_value_encoder: bool = False,
        max_action_dim: int = 512,
        action_cov_rank: int = 16,
        cov_mode: str = "head",
        min_diag_std: float = 1e-4,
        log_std_init: float = 0.0,
        fix_std: bool = False,
        action_granulation: str = "none",
        action_groupings: Optional[Dict[str, MuscleGrouping]] = None,
        action_signatures_by_expert: Optional[Dict[str, List[Tuple[str, ...]]]] = None,
    ):
        """
        Args:
            groups: List[BodyGroup] (single-expert) или
                    Dict[expert_name → List[BodyGroup]] (multi-expert)
            cov_mode: "action_out" — W=action_out, latent_dim=embed_dim
                      "head" — W=projection(action_out), latent_dim=action_cov_rank
            log_std: [1, max_action_dim + latent_dim] как в Lattice
            action_granulation: "none" | "anatomical" | "functional"
            action_groupings: expert_name → MuscleGrouping (если granulation != "none")
            action_signatures_by_expert: expert_name → action signatures (если granulation != "none")
        """
        super().__init__()

        self.vocab = vocab
        self.embed_dim = embed_dim
        self.detached_value_encoder = detached_value_encoder
        self.max_action_dim = max_action_dim
        self.action_cov_rank = action_cov_rank
        self.cov_mode = cov_mode
        self.min_diag_std = min_diag_std

        if action_cov_rank == 0:
            self.latent_dim = 0
        elif cov_mode == "action_out":
            self.latent_dim = embed_dim
        else:
            self.latent_dim = action_cov_rank

        self.obs_normalizer = SignatureNormalizerModule()

        self.body_tokenizer = BodyTokenizer(groups, history_len, embed_dim)

        # Action Tokenizer (грануляция action-side)
        if action_granulation != "none" and action_groupings is not None:
            self.action_tokenizer = ActionTokenizer(
                groupings=action_groupings,
                action_signatures=action_signatures_by_expert,
                embed_dim=embed_dim,
            )
            logger.info(
                f"Action granulation enabled: strategy={action_granulation}, "
                f"decoder queries reduced to group-level"
            )
        else:
            self.action_tokenizer = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True,
        )
        encoder_norm = nn.LayerNorm(embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_enc_layers, norm=encoder_norm)

        action_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True,
        )
        action_decoder_norm = nn.LayerNorm(embed_dim)
        self.action_decoder = nn.TransformerDecoder(
            action_decoder_layer, num_act_dec_layers, norm=action_decoder_norm
        )

        value_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True,
        )
        value_decoder_norm = nn.LayerNorm(embed_dim)
        self.value_decoder = nn.TransformerDecoder(
            value_decoder_layer, num_val_dec_layers, norm=value_decoder_norm
        )

        self.action_mean_head = nn.Linear(embed_dim, 1)
        self.value_head = nn.Linear(embed_dim, 1)
        self.value_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.log_std = nn.Parameter(
            torch.ones(1, max_action_dim + self.latent_dim) * log_std_init,
            requires_grad=not fix_std,
        )

        if cov_mode == "head" and action_cov_rank > 0:
            self.action_factor_head = nn.Linear(embed_dim, action_cov_rank)
        else:
            self.action_factor_head = None

        self.profiler: Optional[SamplingProfiler] = None

        self._init_weights()

        nn.init.orthogonal_(self.action_mean_head.weight, gain=0.1)
        nn.init.zeros_(self.action_mean_head.bias)
        if self.action_factor_head is not None:
            nn.init.orthogonal_(self.action_factor_head.weight, gain=0.05)
            nn.init.zeros_(self.action_factor_head.bias)
        if self.action_tokenizer is not None:
            for head in self.action_tokenizer.expand_heads.values():
                nn.init.orthogonal_(head.weight, gain=0.1)
                nn.init.zeros_(head.bias)

    def enable_profiling(self, profiler: SamplingProfiler) -> None:
        self.profiler = profiler

    def disable_profiling(self) -> None:
        self.profiler = None

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List[Tuple[str, ...]],
        action_signatures: List[Tuple[str, ...]],
        expert_name: str,
        return_std: bool = True,
        return_value: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        p = self.profiler
        batch_size = obs_timeseries.shape[0]

        if p: p.tick("  normalizer")
        obs_timeseries = self.obs_normalizer(obs_signatures, obs_timeseries)
        if p: p.tock("  normalizer")

        if p: p.tick("  body_tokenizer")
        sensory_emb = self.body_tokenizer.encode(obs_timeseries, expert_name=expert_name)
        if p: p.tock("  body_tokenizer")

        if p: p.tick("  vocab_embed_obs")
        group_sigs = self.body_tokenizer.get_group_signatures(expert_name)
        role_emb = self.vocab.get_embedding_batch(group_sigs)
        role_emb = role_emb.unsqueeze(0).expand(batch_size, -1, -1)
        sensory_emb = sensory_emb + role_emb
        if p: p.tock("  vocab_embed_obs")

        if p: p.tick("  transformer_encoder")
        encoder_out = self.encoder(sensory_emb)
        if p: p.tock("  transformer_encoder")

        if self.action_tokenizer is not None:
            # Granulated path: G group queries instead of A muscle queries
            if p: p.tick("  vocab_embed_act_groups")
            act_group_sigs = self.action_tokenizer.get_group_signatures(expert_name)
            group_query = self.vocab.get_embedding_batch(act_group_sigs)
            group_query = group_query.unsqueeze(0).expand(batch_size, -1, -1)
            if p: p.tock("  vocab_embed_act_groups")

            if p: p.tick("  action_decoder")
            group_out = self.action_decoder(group_query, encoder_out)
            if p: p.tock("  action_decoder")

            if p: p.tick("  action_tokenizer")
            mean, action_out = self.action_tokenizer.decode(group_out, expert_name)
            if p: p.tock("  action_tokenizer")
        else:
            # Original path: per-muscle queries
            if p: p.tick("  vocab_embed_act")
            action_query = self.vocab.get_embedding_batch(action_signatures)
            action_query = action_query.unsqueeze(0).expand(batch_size, -1, -1)
            if p: p.tock("  vocab_embed_act")

            if p: p.tick("  action_decoder")
            action_out = self.action_decoder(action_query, encoder_out)
            mean = self.action_mean_head(action_out).squeeze(-1)
            if p: p.tock("  action_decoder")

        cov_factor = None
        diag_std = None
        if return_std:
            num_actions = action_out.shape[1]
            std = torch.nn.functional.softplus(self.log_std) + self.min_diag_std
            diag_std = std[:, :num_actions].expand(batch_size, -1)

            if self.latent_dim == 0:
                cov_factor = torch.zeros(batch_size, num_actions, 1, device=mean.device, dtype=mean.dtype)
            else:
                latent_std = std[:, self.max_action_dim: self.max_action_dim + self.latent_dim].squeeze(0)

                if self.cov_mode == "action_out":
                    cov_factor = action_out * latent_std.unsqueeze(0).unsqueeze(0)
                    cov_factor = cov_factor / (self.latent_dim ** 0.5)
                else:
                    W = self.action_factor_head(action_out)
                    cov_factor = W * latent_std.unsqueeze(0).unsqueeze(0)

        value = None
        if return_value:
            if p: p.tick("  value_decoder")
            value_query = self.value_query.expand(batch_size, -1, -1)
            value_encoder_out = encoder_out.detach() if self.detached_value_encoder else encoder_out
            value_out = self.value_decoder(value_query, value_encoder_out)
            value = self.value_head(value_out).squeeze(-1)
            if p: p.tock("  value_decoder")

        return mean, cov_factor, diag_std, value

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
        mean, cov_factor, diag_std, value = self.forward(
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
            log_prob = dist.log_prob(action).unsqueeze(-1)

        return action, log_prob, value

    def get_log_prob(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List[Tuple[str, ...]],
        action_signatures: List[Tuple[str, ...]],
        actions: torch.Tensor,
        expert_name: Optional[str] = None,
    ) -> torch.Tensor:
        mean, cov_factor, diag_std, value = self.forward(
            obs_timeseries,
            obs_signatures,
            action_signatures,
            expert_name=expert_name or "",
        )
        return self._compute_log_prob(actions, mean, cov_factor, diag_std)

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
        return dist.log_prob(actions.float()).unsqueeze(-1)
