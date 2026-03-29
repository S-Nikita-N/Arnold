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
import torch.nn.functional as F
from torch.distributions import LowRankMultivariateNormal
from typing import Any, List, Dict, Optional, Tuple, Union
import logging

from arnold.torch_model.sensorimotor_vocabulary import SensorimotorVocabulary
from arnold.torch_model.normalization import SignatureNormalizerModule
from arnold.torch_model.body_tokenizer import BodyTokenizer
from arnold.torch_model.action_tokenizer import ActionTokenizer, GroupMuscleMap, SizeBucket
from arnold.action_parser import MuscleGrouping
from arnold.observation_parser import BodyGroup
from arnold.profiler import SamplingProfiler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hierarchical Action Decoder layers
# ---------------------------------------------------------------------------

class GroupToGroupLayer(nn.Module):
    """Pre-norm transformer layer для group tokens с optional cross-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        cross_attention: bool,
    ):
        super().__init__()
        self.cross_attention = cross_attention

        # Self-attention
        self.sa_norm = nn.LayerNorm(embed_dim)
        self.sa = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention (optional)
        if cross_attention:
            self.ca_norm = nn.LayerNorm(embed_dim)
            self.ca = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        # FFN
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        group_tokens: torch.Tensor,
        encoder_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention (pre-norm)
        x = self.sa_norm(group_tokens)
        group_tokens = (
            group_tokens + 
            self.sa(x, x, x, need_weights=False)[0]
        )  # [B, G, D]

        # Cross-attention (pre-norm)
        if self.cross_attention and encoder_out is not None:
            group_tokens = (
                group_tokens + 
                self.ca(
                    self.ca_norm(group_tokens),
                    encoder_out,
                    encoder_out,
                    need_weights=False
                )[0]  # [B, G, D]
            )

        # FFN (pre-norm)
        group_tokens = (
            group_tokens + 
            self.ff(self.ff_norm(group_tokens))
        )
        return group_tokens


class WithinGroupLayer(nn.Module):
    """Pre-norm transformer layer с batched SDPA внутри каждой группы.

    Для каждой группы g собирается sequence [group_token_g, muscle_1, ..., muscle_k],
    все группы паддятся до max_group_size+1 и обрабатываются как один batch через SDPA.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        cross_attention: bool,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.cross_attention = cross_attention

        # Self-attention (manual QKV для batched SDPA)
        self.sa_norm = nn.LayerNorm(embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = dropout

        # Cross-attention (optional)
        if cross_attention:
            self.ca_norm = nn.LayerNorm(embed_dim)
            self.ca = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        # FFN
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        group_tokens: torch.Tensor,
        muscle_tokens: torch.Tensor,
        gm_map: GroupMuscleMap,
        encoder_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, G, D = group_tokens.shape
        head_dim = D // self.num_heads
        dropout_p = self.attn_dropout if self.training else 0.0

        group_tokens_out = group_tokens.clone()
        muscle_tokens_out = muscle_tokens.clone()

        for bucket in gm_map.buckets:
            n_b = bucket.group_indices.shape[0]
            max_gs = bucket.max_size
            seq_len = max_gs + 1

            # Gather group tokens for this bucket: [B, n_b, D]
            b_group = group_tokens[:, bucket.group_indices]
            # Gather muscle tokens: [B, n_b, max_gs, D]
            b_muscles = muscle_tokens[:, bucket.gather_idx]

            # Assemble sequence: [group_token, muscle_1, ..., muscle_k, PAD...]
            x = torch.cat(
                [
                    b_group.unsqueeze(2),
                    b_muscles
                ],
                dim=2,
            ).reshape(B * n_b, seq_len, D)

            # --- Self-attention with pre-computed mask ---
            qkv = self.qkv_proj(self.sa_norm(x)).reshape(
                B * n_b, seq_len, 3, self.num_heads, head_dim
            )
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)  # [B*n_b, H, seq_len, head_dim]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Expand pre-computed mask: [n_b, 1, seq_len, seq_len] → [B*n_b, 1, seq_len, seq_len]
            attn_mask = bucket.attn_mask.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(
                B * n_b, 1, seq_len, seq_len
            )

            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
            attn_out = attn_out.transpose(1, 2).reshape(B * n_b, seq_len, D)
            x = x + self.out_proj(attn_out)

            # --- Cross-attention (optional) ---
            if self.cross_attention and encoder_out is not None:
                enc_expanded = (
                    encoder_out.unsqueeze(1)
                    .expand(-1, n_b, -1, -1)
                    .reshape(B * n_b, -1, D)
                )
                x = x + self.ca(self.ca_norm(x), enc_expanded, enc_expanded, need_weights=False)[0]

            # --- FFN ---
            x = x + self.ff(self.ff_norm(x))

            # --- Scatter back ---
            x = x.reshape(B, n_b, seq_len, D)
            group_tokens_out[:, bucket.group_indices] = x[:, :, 0, :]

            # Vectorized scatter muscles
            b_muscles_out = x[:, :, 1:, :]  # [B, n_b, max_gs, D]
            valid = ~bucket.pad_mask[:, 1:]  # [n_b, max_gs] — True = valid
            tgt_idx = bucket.gather_idx[valid]  # [n_valid]
            muscle_tokens_out[:, tgt_idx] = b_muscles_out[:, valid]

        return group_tokens_out, muscle_tokens_out


class HierarchicalActionDecoder(nn.Module):
    """Configurable stack of group_to_group и within_group слоёв."""

    def __init__(
        self,
        layer_configs: List[Dict[str, Any]],
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()

        layers = []
        for cfg in layer_configs:
            layer_type = cfg["type"]
            cross_attn = cfg.get("cross_attn", layer_type == "g2g")
            if layer_type == "g2g":
                layers.append(
                    GroupToGroupLayer(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        ff_dim=ff_dim,
                        dropout=dropout,
                        cross_attention=cross_attn,
                    )
                )
            elif layer_type == "wg":
                layers.append(
                    WithinGroupLayer(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        ff_dim=ff_dim,
                        dropout=dropout,
                        cross_attention=cross_attn,
                    )
                )
            else:
                raise ValueError(f"Unknown layer type: {layer_type!r} (use 'g2g' or 'wg')")

        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.LayerNorm(embed_dim)

        self.has_within_group = any(
            isinstance(layer, WithinGroupLayer)
            for layer in self.layers
        )

        types_str = " → ".join(
            ("G2G" if isinstance(layer, GroupToGroupLayer) else "WG")
            + ("+CA" if layer.cross_attention else "")
            for layer in self.layers
        )
        logger.info(
            f"HierarchicalActionDecoder: {len(self.layers)} layers [{types_str}]"
        )

    def forward(
        self,
        group_tokens: torch.Tensor,
        encoder_out: torch.Tensor,
        muscle_query: Optional[torch.Tensor],
        gm_map: Optional[GroupMuscleMap],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            group_tokens: [B, G, D] — group query embeddings
            encoder_out: [B, S, D] — encoder output
            muscle_query: [B, A, D] — per-muscle vocab embeddings в grouped order (None если нет within_group)
            gm_map: GroupMuscleMap для within_group attention

        Returns:
            group_out: [B, G, D]
            muscle_out: [B, A, D] в grouped order, или None если нет within_group слоёв
        """
        muscle_tokens: Optional[torch.Tensor] = None

        for layer in self.layers:
            if isinstance(layer, GroupToGroupLayer):
                group_tokens = layer(group_tokens, encoder_out)
            elif isinstance(layer, WithinGroupLayer):
                if muscle_tokens is None:
                    muscle_tokens = muscle_query

                group_tokens, muscle_tokens = layer(
                    group_tokens,
                    muscle_tokens,
                    gm_map,
                    encoder_out,
                )

        group_tokens = self.final_norm(group_tokens)
        if muscle_tokens is not None:
            muscle_tokens = self.final_norm(muscle_tokens)

        return group_tokens, muscle_tokens


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
        action_decoder_layers: Optional[List[Dict[str, Any]]] = None,
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
                strategy=action_granulation,
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

        # Action decoder: hierarchical (if configured) or legacy uniform
        if action_decoder_layers is not None:
            self.hierarchical_decoder = HierarchicalActionDecoder(
                layer_configs=action_decoder_layers,
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            self.action_decoder = None
        else:
            self.hierarchical_decoder = None
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
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

            if self.hierarchical_decoder is not None:
                # Hierarchical path: configurable layer stack
                muscle_query = None
                if self.hierarchical_decoder.has_within_group:
                    if p: p.tick("  vocab_embed_muscles")
                    muscle_sigs = self.action_tokenizer.get_muscle_signatures(expert_name)
                    muscle_query = self.vocab.get_embedding_batch(muscle_sigs)
                    muscle_query = muscle_query.unsqueeze(0).expand(batch_size, -1, -1)
                    if p: p.tock("  vocab_embed_muscles")

                if p: p.tick("  action_decoder")
                gm_map = self.action_tokenizer.get_group_muscle_map(expert_name)
                group_out, muscle_out = self.hierarchical_decoder(
                    group_query,
                    encoder_out,
                    muscle_query,
                    gm_map,
                )
                if p: p.tock("  action_decoder")

                if muscle_out is not None:
                    # within_group слои были — muscle_out в grouped order, reorder
                    if p: p.tick("  action_tokenizer")
                    reorder_idx = getattr(self.action_tokenizer, f"reorder_idx_{expert_name}")
                    D = muscle_out.shape[-1]
                    action_out = torch.gather(
                        muscle_out, 1,
                        reorder_idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, D),
                    )
                    mean = self.action_mean_head(action_out).squeeze(-1)
                    if p: p.tock("  action_tokenizer")
                else:
                    # Только group_to_group слои — fallback на expand_heads
                    if p: p.tick("  action_tokenizer")
                    mean, action_out = self.action_tokenizer.decode(group_out, expert_name)
                    if p: p.tock("  action_tokenizer")
            else:
                # Legacy path: uniform nn.TransformerDecoder
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
        latent_std = None
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

        return mean, cov_factor, diag_std, value, latent_std

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
        mean, cov_factor, diag_std, value, _ = self.forward(
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
        mean, cov_factor, diag_std, value, _ = self.forward(
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
