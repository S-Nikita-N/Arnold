"""
Arnold Transformer Policy.

Encoder-Decoder Transformer для управления мышцами.
- Body Tokenizer группирует наблюдения по телам (~70 токенов вместо ~953)
- Shared Encoder обрабатывает body-level embeddings
- Action Decoder генерирует muscle activations
- Value Decoder генерирует state value
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from arnold.torch_model.sensorimotor_vocabulary import SensorimotorVocabulary
from arnold.torch_model.normalization import SignatureNormalizerModule
from arnold.torch_model.body_tokenizer import BodyTokenizer
from arnold.observation_parser import BodyGroup
from arnold.profiler import SamplingProfiler


class TransformerPolicy(nn.Module):
    """
    Arnold Transformer Policy с body-level tokenization.

    Вместо 953 скалярных токенов (один на элемент наблюдения)
    используется ~70 body-level токенов. Self-attention: 70² vs 953² = 185x быстрее.
    """

    def __init__(
        self,
        vocab: SensorimotorVocabulary,
        groups: List[BodyGroup],
        history_len: int = 5,
        embed_dim: int = 128,
        ff_dim: int = 512,
        num_heads: int = 4,
        num_enc_layers: int = 6,
        num_act_dec_layers: int = 6,
        num_val_dec_layers: int = 1,
        dropout: float = 0.0,
        detached_value_encoder: bool = False,
    ):
        super().__init__()

        self.vocab = vocab
        self.embed_dim = embed_dim
        self.detached_value_encoder = detached_value_encoder

        self.obs_normalizer = SignatureNormalizerModule()

        # Body tokenizer: flat obs → body-level tokens
        self.body_tokenizer = BodyTokenizer(groups, history_len, embed_dim)

        # Transformer Encoder
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

        # Action Decoder
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

        # Value Decoder
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

        # Output heads
        self.action_mean_head = nn.Linear(embed_dim, 1)
        self.action_std_head = nn.Linear(embed_dim, 1)
        self.value_head = nn.Linear(embed_dim, 1)

        self.log_sigma_global = nn.Parameter(torch.zeros(1))
        self.value_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.profiler: Optional[SamplingProfiler] = None

        self._init_weights()

        # Инициализируем action_mean около нуля, чтобы не было изначального взрыва
        nn.init.orthogonal_(self.action_mean_head.weight, gain=0.1)
        nn.init.zeros_(self.action_mean_head.bias)
        
        # # Инициализируем std независимым от состояния, около -0.5 (std ~ 0.6)
        # # Это даст нормальное начальное исследование (exploration) без softmax
        # nn.init.zeros_(self.action_std_head.weight)
        # nn.init.constant_(self.action_std_head.bias, -0.5)

    def enable_profiling(self, profiler: SamplingProfiler) -> None:
        self.profiler = profiler

    def disable_profiling(self) -> None:
        self.profiler = None

    def _init_weights(self):
        """Xavier инициализация для всех Linear слоев."""
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
        return_std: bool = True,
        return_value: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            obs_timeseries: [batch, n_obs_flat, history_len] — flat наблюдения
            obs_signatures: per-scalar сигнатуры (для нормализатора)
            action_signatures: сигнатуры для мышц
            return_std: если False — не считать std head
            return_value: если False — не считать value decoder

        Returns:
            actions: [batch, num_actions]
            log_std: [batch, num_actions] или None
            value: [batch, 1] или None
        """
        p = self.profiler
        batch_size = obs_timeseries.shape[0]

        # 1. Per-scalar нормализация (obs_signatures нужны нормализатору)
        if p: p.tick("  normalizer")
        obs_timeseries = self.obs_normalizer(obs_signatures, obs_timeseries)
        if p: p.tock("  normalizer")

        # 2. Body tokenizer: [B, n_obs_flat, history_len] → [B, n_tokens, embed_dim]
        if p: p.tick("  body_tokenizer")
        sensory_emb = self.body_tokenizer(obs_timeseries)
        if p: p.tock("  body_tokenizer")

        # 3. Role embeddings для body-level групп
        if p: p.tick("  vocab_embed_obs")
        role_emb = self.vocab.get_embedding_batch(self.body_tokenizer.group_signatures)
        role_emb = role_emb.unsqueeze(0).expand(batch_size, -1, -1)
        sensory_emb = sensory_emb + role_emb
        if p: p.tock("  vocab_embed_obs")

        # 4. Transformer Encoder
        if p: p.tick("  transformer_encoder")
        encoder_out = self.encoder(sensory_emb)
        if p: p.tock("  transformer_encoder")

        # 5. Action queries из vocabulary
        if p: p.tick("  vocab_embed_act")
        action_query = self.vocab.get_embedding_batch(action_signatures)
        action_query = action_query.unsqueeze(0).expand(batch_size, -1, -1)
        if p: p.tock("  vocab_embed_act")

        # 6. Action Decoder
        if p: p.tick("  action_decoder")
        action_out = self.action_decoder(action_query, encoder_out)
        actions = self.action_mean_head(action_out).squeeze(-1)
        if p: p.tock("  action_decoder")

        log_std = None
        if return_std:
            num_actions = action_out.shape[1]
            std_logits = self.action_std_head(action_out).squeeze(-1)
            log_soft = torch.log_softmax(std_logits, dim=-1)
            log_sigma_global = self.log_sigma_global.view(1, 1)
            log_norm_factor = torch.log(torch.tensor(
                num_actions, dtype=log_soft.dtype, device=log_soft.device, requires_grad=False
            ))
            log_std = log_sigma_global + log_soft + log_norm_factor
            log_std = torch.clamp(log_std, min=-4.6, max=2.3)

        # 7. Value Decoder
        value = None
        if return_value:
            if p: p.tick("  value_decoder")
            value_query = self.value_query.expand(batch_size, -1, -1)
            value_encoder_out = encoder_out.detach() if self.detached_value_encoder else encoder_out
            value_out = self.value_decoder(value_query, value_encoder_out)
            value = self.value_head(value_out).squeeze(-1)
            if p: p.tock("  value_decoder")

        return actions, log_std, value

    def get_action(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List[Tuple[str, ...]],
        action_signatures: List[Tuple[str, ...]],
        deterministic: bool = False,
        return_std: bool = True,
        return_value: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Получает действие для среды.

        Args:
            obs_timeseries: [batch, n_obs, history_len]
            obs_signatures: per-scalar сигнатуры
            action_signatures: сигнатуры мышц
            deterministic: если True — без шума
            return_std: если False — не считать std head
            return_value: если False — не считать value decoder

        Returns:
            action: [batch, num_muscles]
            log_prob: [batch, 1] или None при return_std=False
            value: [batch, 1] или None при return_value=False
        """
        mean, log_std, value = self.forward(
            obs_timeseries,
            obs_signatures,
            action_signatures,
            return_std=return_std,
            return_value=return_value,
        )

        if deterministic:
            action = mean
            log_prob = None if return_std else torch.zeros(mean.shape[0], 1, device=mean.device)
        else:
            if log_std is None:
                raise ValueError("return_std=False допустим только при deterministic=True")
            action = self.sample_action(mean, log_std)
            log_prob = self._compute_log_prob(action, mean, log_std)

        return action, log_prob, value

    def sample_action(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        std = log_std.exp()
        noise = torch.randn_like(mean)
        return mean + noise * std

    def get_log_prob(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List[Tuple[str, ...]],
        action_signatures: List[Tuple[str, ...]],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Вычисляет log probability для заданных действий.
        """
        mean, log_std, _ = self.forward(
            obs_timeseries,
            obs_signatures,
            action_signatures
        )
        return self._compute_log_prob(actions, mean, log_std)

    def _compute_log_prob(
        self,
        actions: torch.Tensor,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        log N(a | μ, σ) = -0.5 * (log(2π) + 2*log(σ) + ((a-μ)/σ)²)
        """
        var = (log_std * 2).exp()
        log_prob = -0.5 * (
            torch.log(torch.tensor(2 * 3.14159265359, device=actions.device, dtype=actions.dtype)) +
            2 * log_std +
            (actions - mean).pow(2) / var
        )
        return log_prob.sum(dim=-1, keepdim=True)
