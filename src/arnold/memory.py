"""
Memory и Batch для OBC training.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class OBCMemory:
    """Хранит данные для OBC обновления."""
    states: List[torch.Tensor] = field(default_factory=list)       # [n_obs, history_len]
    obs_signatures: List[List[Tuple[str, ...]]] = field(default_factory=list)
    action_signatures: List[List[Tuple[str, ...]]] = field(default_factory=list)
    student_actions: List[torch.Tensor] = field(default_factory=list)
    expert_actions: List[torch.Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[torch.Tensor] = field(default_factory=list)
    masks: List[float] = field(default_factory=list)  # 0 если episode done
    log_probs: List[torch.Tensor] = field(default_factory=list)  # log prob действий
    
    def clear(self):
        self.states.clear()
        self.obs_signatures.clear()
        self.action_signatures.clear()
        self.student_actions.clear()
        self.expert_actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.masks.clear()
        self.log_probs.clear()
    
    def extend(self, other: "OBCMemory"):
        """Расширяет memory данными из другого memory."""
        self.states.extend(other.states)
        self.obs_signatures.extend(other.obs_signatures)
        self.action_signatures.extend(other.action_signatures)
        self.student_actions.extend(other.student_actions)
        self.expert_actions.extend(other.expert_actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.masks.extend(other.masks)
        self.log_probs.extend(other.log_probs)
    
    def __len__(self):
        return len(self.states)

    def to_batch(
        self,
        gamma: float,
        tau: float,
        device: Optional[torch.device] = None,
    ) -> "OBCBatch":
        """
        Конвертирует OBCMemory в OBCBatch и считает GAE.

        Args:
            gamma: discount factor
            tau:   GAE lambda
            device: устройство для тензоров (если None — остаёмся на CPU)
        """
        if len(self.states) == 0:
            raise ValueError("OBCMemory is empty, cannot convert to batch.")

        # Все signatures одинаковые (структура env фиксирована)
        obs_signatures = self.obs_signatures[0]
        action_signatures = self.action_signatures[0]

        # Stack tensors
        states = torch.stack(self.states, dim=0)
        student_actions = torch.stack(self.student_actions, dim=0)
        expert_actions = torch.stack(self.expert_actions, dim=0)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.stack(self.values, dim=0)
        masks = torch.tensor(self.masks, dtype=torch.float32)
        log_probs = torch.stack(self.log_probs, dim=0)  # fixed log_probs

        # --- GAE ---
        rewards_exp = rewards.unsqueeze(-1)
        masks_exp = masks.unsqueeze(-1)

        batch_size = rewards_exp.size(0)
        deltas = torch.zeros(batch_size, 1)
        advantages = torch.zeros(batch_size, 1)

        prev_value = 0.0
        prev_advantage = 0.0

        for i in reversed(range(batch_size)):
            # TD error: δ_t = r_t + γ * V(s_{t+1}) * mask - V(s_t)
            deltas[i] = rewards_exp[i] + gamma * prev_value * masks_exp[i] - values[i]
            # GAE: A_t = δ_t + γ * λ * A_{t+1} * mask
            advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks_exp[i]

            prev_value = values[i, 0].item()
            prev_advantage = advantages[i, 0].item()

        # Returns = V + A
        returns = values + advantages

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if device is not None:
            states = states.to(device)
            student_actions = student_actions.to(device)
            expert_actions = expert_actions.to(device)
            rewards = rewards.to(device)
            values = values.to(device)
            masks = masks.to(device)
            log_probs = log_probs.to(device)
            returns = returns.to(device)
            advantages = advantages.to(device)

        return OBCBatch(
            states=states,
            obs_signatures=obs_signatures,
            action_signatures=action_signatures,
            student_actions=student_actions,
            expert_actions=expert_actions,
            rewards=rewards,
            values=values,
            masks=masks,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        )


@dataclass 
class OBCBatch:
    """Батч для OBC обновления."""
    states: torch.Tensor           # [batch, n_obs, history_len]
    obs_signatures: List[Tuple[str, ...]]
    action_signatures: List[Tuple[str, ...]]
    student_actions: torch.Tensor  # [batch, n_actions]
    expert_actions: torch.Tensor   # [batch, n_actions]
    rewards: torch.Tensor          # [batch]
    values: torch.Tensor           # [batch, 1]
    masks: torch.Tensor            # [batch]
    log_probs: torch.Tensor        # [batch, 1] - log prob действий (fixed, до обновления)
    returns: torch.Tensor = None   # [batch, 1] - GAE returns
    advantages: torch.Tensor = None  # [batch, 1] - GAE advantages
