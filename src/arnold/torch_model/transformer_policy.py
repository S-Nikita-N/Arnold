"""
Arnold Transformer Policy.

Encoder-Decoder Transformer для управления мышцами.
- Shared Encoder обрабатывает sensory embeddings
- Action Decoder генерирует muscle activations
- Value Decoder генерирует state value
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from myohuman.arnold.torch_model.sensorimotor_vocabulary import SensorimotorVocabulary
from myohuman.arnold.torch_model.normalization import SignatureNormalizerModule
from myohuman.arnold.torch_model.sensory_encoder import SensoryEncoder


class TransformerPolicy(nn.Module):
    """
    Arnold Transformer Policy.
    
    - Encoder: 6 layers, 128 dim, 512 ff, 4 heads
    - Action Decoder: 6 layers, 128 dim, 512 ff, 4 heads
    - Value Decoder: 6 layers, 128 dim, 512 ff, 4 heads
    """
    
    def __init__(
        self,
        vocab: SensorimotorVocabulary,
        history_len: int = 5,
        embed_dim: int = 128,
        ff_dim: int = 512,
        num_heads: int = 4,
        num_layers: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.vocab = vocab
        self.embed_dim = embed_dim

        self.obs_normalizer = SignatureNormalizerModule()
        
        # Sensory encoder: time series → embedding
        self.sensory_encoder = SensoryEncoder(history_len, embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True,  # Pre-norm
        )
        # При norm_first=True (Pre-LN) выход encoder ненормализован!
        # Добавляем финальный LayerNorm для стабилизации.
        encoder_norm = nn.LayerNorm(embed_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=encoder_norm)
        
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
        self.action_decoder = nn.TransformerDecoder(action_decoder_layer, num_layers, norm=action_decoder_norm)
        
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
        self.value_decoder = nn.TransformerDecoder(value_decoder_layer, num_layers, norm=value_decoder_norm)
        
        # Output heads
        self.action_mean_head = nn.Linear(embed_dim, 1)
        self.action_std_head = nn.Linear(embed_dim, 1)
        self.value_head = nn.Linear(embed_dim, 1)

        # Глобальный learnable scalar для шума (sigma_global)
        self.log_sigma_global = nn.Parameter(torch.zeros(1))
        
        # Value query embedding (learnable)
        self.value_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier инициализация для всех Linear слоев."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_observations(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List[Tuple[str, ...]],
    ) -> torch.Tensor:
        """
        Кодирует наблюдения в sensory embeddings.
        
        Args:
            obs_timeseries: [batch, n_obs, history_len] - временные ряды
            obs_signatures: список кортежей токенов для каждого observation element
        
        Returns:
            [batch, n_obs, embed_dim] - sensory embeddings с role embeddings
        """
        batch_size = obs_timeseries.shape[0]
        
        # Time series → embedding
        sensory_emb = self.sensory_encoder(obs_timeseries)  # [batch, n_obs, embed_dim]
        
        # Role embeddings из vocabulary
        role_emb = self.vocab.get_embedding_batch(obs_signatures)  # [n_obs, embed_dim]
        role_emb = role_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, n_obs, embed_dim]
        
        # Суммируем
        return sensory_emb + role_emb
    
    def forward(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List[Tuple[str, ...]],
        action_signatures: List[Tuple[str, ...]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs_timeseries: [batch, n_obs, history_len] - наблюдения
            obs_signatures: список токенов для каждого obs element
            action_signatures: список токенов для каждой мышцы
        
        Returns:
            actions: [batch, num_actions] - mean actions
            log_std: [batch, num_actions] - log std для exploration
            value: [batch, 1] - state value
        """
        batch_size = obs_timeseries.shape[0]

        # 1. Нормализация наблюдений (обновление статистик происходит внутри forward)
        obs_timeseries = self.obs_normalizer(obs_signatures, obs_timeseries)
        
        # 2. Encode observations
        sensory_emb = self.encode_observations(obs_timeseries, obs_signatures)
        
        # 3. Transformer Encoder
        encoder_out = self.encoder(sensory_emb)  # [batch, n_obs, embed_dim]
        
        # 4. Action Decoder
        # Query = muscle embeddings
        action_query = self.vocab.get_embedding_batch(action_signatures)  # [n_muscles, embed_dim]
        action_query = action_query.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, n_muscles, embed_dim]
        action_out = self.action_decoder(action_query, encoder_out)  # [batch, n_actions, embed_dim]
        actions = self.action_mean_head(action_out).squeeze(-1)  # [batch, n_actions]
        # Std (task-specific exploratory noise): sigma = sigma_global * softmax(E @ x) * N_A
        num_actions = action_out.shape[1]
        std_logits = self.action_std_head(action_out).squeeze(-1)          # [batch, n_actions]
        log_soft = torch.log_softmax(std_logits, dim=-1)                   # [batch, n_actions]
        log_sigma_global = self.log_sigma_global.view(1, 1)                # [1,1]
        log_norm_factor = torch.log(torch.tensor(
            num_actions, dtype=log_soft.dtype, device=log_soft.device, requires_grad=False
        ))
        log_std = log_sigma_global + log_soft + log_norm_factor  # [batch, n_actions]
        log_std = torch.clamp(log_std, min=-4.6, max=2.3)
        
        # 5. Value Decoder
        value_query = self.value_query.expand(batch_size, -1, -1) # [batch, 1, embed_dim]
        value_out = self.value_decoder(value_query, encoder_out)  # [batch, 1, embed_dim]
        value = self.value_head(value_out).squeeze(-1)  # [batch, 1]
        
        return actions, log_std, value
    
    def get_action(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List[Tuple[str, ...]],
        action_signatures: List[Tuple[str, ...]],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Получает действие для среды.
        
        Args:
            obs_timeseries: [batch, n_obs, history_len]
            obs_signatures: список токенов для obs
            action_signatures: список токенов для actions
            deterministic: если True - без шума
        
        Returns:
            action: [batch, num_muscles]
            log_prob: [batch, 1] - log probability действия
            value: [batch, 1]
        """
        mean, log_std, value = self.forward(
            obs_timeseries,
            obs_signatures,
            action_signatures
        )
        
        if deterministic:
            # Для deterministic действия log_prob не имеет смысла
            action = mean   
            log_prob = torch.zeros(mean.shape[0], 1, device=mean.device)
        else:
            action = self.sample_action(mean, log_std)
            log_prob = self._compute_log_prob(action, mean, log_std)
        
        return action, log_prob, value

    def sample_action(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample action from the policy.
        """
        std = log_std.exp()
        noise = torch.randn_like(mean)
        action = mean + noise * std
        return action

    def get_log_prob(
        self,
        obs_timeseries: torch.Tensor,
        obs_signatures: List[Tuple[str, ...]],
        action_signatures: List[Tuple[str, ...]],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Вычисляет log probability для заданных действий.
        
        Args:
            obs_timeseries: [batch, n_obs, history_len]
            obs_signatures: список токенов для obs
            action_signatures: список токенов для actions
            actions: [batch, num_muscles] - действия для оценки
        
        Returns:
            log_prob: [batch, 1]
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
        Вычисляет log probability под diagonal Gaussian.
        
        log N(a | μ, σ) = -0.5 * (log(2π) + 2*log(σ) + ((a-μ)/σ)²)
        
        Args:
            actions: [batch, n_actions]
            mean: [batch, n_actions]
            log_std: [batch, n_actions]
        
        Returns:
            log_prob: [batch, 1] - сумма по всем action dimensions
        """
        var = (log_std * 2).exp()
        log_prob = -0.5 * (
            torch.log(torch.tensor(2 * 3.14159265359, device=actions.device, dtype=actions.dtype)) + 
            2 * log_std + 
            (actions - mean).pow(2) / var
        )
        # Сумма по action dimensions
        return log_prob.sum(dim=-1, keepdim=True)


if __name__ == "__main__":
    # Quick test
    vocab = SensorimotorVocabulary(embed_dim=128)
    policy = TransformerPolicy(vocab)
    policy.train()
    # Dummy data
    batch_size = 4
    n_obs = 100
    history_len = 5
    
    obs = torch.randn(batch_size, n_obs, history_len)
    obs_sigs = [("femur", "l", "position", "x")] * n_obs
    act_sigs = [("soleus", "r", "muscle", "activation")] * 80
    
    actions, log_std, value = policy(obs, obs_sigs, act_sigs)
    print(f"Actions: {actions.shape}")  # [4, 80]
    print(f"Log std: {log_std.shape}")  # [80]
    print(f"Value: {value.shape}")      # [4, 1]
