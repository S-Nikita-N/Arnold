import torch
import torch.nn as nn


class SensoryEncoder(nn.Module):
    """
    Линейный encoder для временных рядов наблюдений.
    
    5 timesteps → embed_dim через Linear projection.
    """
    
    def __init__(self, history_len: int = 5, embed_dim: int = 128):
        super().__init__()
        self.history_len = history_len
        self.embed_dim = embed_dim
        self.linear = nn.Linear(history_len, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_elements, history_len] - временные ряды
        
        Returns:
            [batch, n_elements, embed_dim] - sensory embeddings
        """
        return self.linear(x)
