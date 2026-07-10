"""
Logger для OBC training.
По мотивам Kinesis LoggerRL.
"""

import math
import numpy as np

from typing import Any
from collections import defaultdict


########################################
#              OBCLogger               #
########################################


class OBCLogger:
    """
    Логгер для OBC обучения.
    Совместим с Kinesis LoggerRL для merge.
    """

    ########################################
    #              Recording               #
    ########################################

    def __init__(self):
        # Episode tracking
        self.num_steps = 0
        self.num_episodes = 0
        self.total_reward = 0.0
        self.episode_reward = 0.0

        # Reward statistics
        self.min_episode_reward = math.inf
        self.max_episode_reward = -math.inf
        self.min_reward = math.inf
        self.max_reward = -math.inf

        # Losses из update_params (ppo, imitation, value, entropy)
        self.ppo_loss = 0.0
        self.imitation_loss = 0.0
        self.value_loss = 0.0
        self.entropy_loss = 0.0
        self.load_balance_loss = 0.0
        self.mean_penalty_loss = 0.0

        # Episode lists
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self._current_episode_length = 0

        # Timing
        self.sample_time = 0.0
        self.update_time = 0.0

        # Info dict (как в Kinesis для компонентов reward)
        self.info_dict: dict[str, list[float]] = defaultdict(list)

    def step(
        self,
        reward: float,
        info: dict[str, Any] = None,
    ):
        """Записывает один шаг."""
        self.episode_reward += reward
        self._current_episode_length += 1
        self.num_steps += 1

        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)

        # Сохраняем компоненты reward из info
        if info:
            for k, v in info.items():
                if isinstance(v, (bool, np.bool_)):
                    self.info_dict[k].append(1.0 if v else 0.0)
                elif isinstance(v, (int, float, np.floating, np.integer)):
                    self.info_dict[k].append(float(v))

    def end_episode(self):
        """Завершает эпизод."""
        self.episode_rewards.append(self.episode_reward)
        self.episode_lengths.append(self._current_episode_length)
        self.total_reward += self.episode_reward
        self.num_episodes += 1

        self.min_episode_reward = min(
            self.min_episode_reward, self.episode_reward
        )
        self.max_episode_reward = max(
            self.max_episode_reward, self.episode_reward
        )

        self.episode_reward = 0.0
        self._current_episode_length = 0

    def end_sampling(self):
        """Вызывается после сбора всех траекторий."""
        pass  # Статистика вычисляется через properties

    ########################################
    #              Properties              #
    ########################################

    @property
    def avg_episode_reward(self) -> float:
        if self.num_episodes == 0:
            return 0.0
        return self.total_reward / self.num_episodes

    @property
    def avg_episode_len(self) -> float:
        if self.num_episodes == 0:
            return 0.0
        return self.num_steps / self.num_episodes

    @property
    def avg_reward(self) -> float:
        """Средний reward за шаг."""
        if self.num_steps == 0:
            return 0.0
        return self.total_reward / self.num_steps

    @property
    def total_loss(self) -> float:
        """Общая сумма лоссов."""
        return (
            self.ppo_loss
            + self.imitation_loss
            + self.value_loss
            + self.entropy_loss
            + self.load_balance_loss
            + self.mean_penalty_loss
        )

    def set_update_losses(
        self,
        ppo_loss: float,
        imitation_loss: float,
        value_loss: float,
        entropy_loss: float,
        load_balance_loss: float,
        mean_penalty_loss: float = 0.0,
    ) -> None:
        """Присваивает лоссы, посчитанные в update_params."""
        self.ppo_loss = ppo_loss
        self.imitation_loss = imitation_loss
        self.value_loss = value_loss
        self.entropy_loss = entropy_loss
        self.load_balance_loss = load_balance_loss
        self.mean_penalty_loss = mean_penalty_loss

    def reset_losses(self):
        """Сбрасывает losses (между эпохами)."""
        self.ppo_loss = 0.0
        self.imitation_loss = 0.0
        self.value_loss = 0.0
        self.entropy_loss = 0.0
        self.load_balance_loss = 0.0
        self.mean_penalty_loss = 0.0

    def reset_episodes(self):
        """Сбрасывает episode статистику."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.info_dict.clear()

    ########################################
    #                Merge                 #
    ########################################

    @classmethod
    def merge(cls, logger_list: list["OBCLogger"]) -> "OBCLogger":
        """Объединяет логгеры из нескольких процессов."""
        merged = cls()

        for lg in logger_list:
            if lg is None:
                continue

            merged.num_steps += lg.num_steps
            merged.num_episodes += lg.num_episodes
            merged.total_reward += lg.total_reward

            merged.min_reward = min(merged.min_reward, lg.min_reward)
            merged.max_reward = max(merged.max_reward, lg.max_reward)
            merged.min_episode_reward = min(
                merged.min_episode_reward, lg.min_episode_reward
            )
            merged.max_episode_reward = max(
                merged.max_episode_reward, lg.max_episode_reward
            )

            merged.episode_rewards.extend(lg.episode_rewards)
            merged.episode_lengths.extend(lg.episode_lengths)

            # Merge info_dict
            for k, v in lg.info_dict.items():
                merged.info_dict[k].extend(v)

        return merged

    ########################################
    #               Logging                #
    ########################################

    def get_log_str(self, epoch: int, expert_name: str) -> str:
        """Возвращает строку для печати."""
        reward_str = " ".join(
            [
                f"{np.mean(v):.3f}"
                for k, v in self.info_dict.items()
                if len(v) > 0
            ]
        )

        return (
            f"Ep: {epoch} \t {expert_name} "
            f"T_s {self.sample_time:.2f} T_u {self.update_time:.2f} \t "
            f"eps_R_avg {self.avg_episode_reward:.4f} "
            f"R_avg {self.avg_reward:.4f} "
            f"R_range ({self.min_reward:.4f}, {self.max_reward:.4f}) "
            f"[{reward_str}] \t "
            f"L_ppo {self.ppo_loss:.4f} L_im {self.imitation_loss:.4f} "
            f"L_val {self.value_loss:.4f} L_ent {self.entropy_loss:.4f} "
            f"L_bal {self.load_balance_loss:.4f} "
            f"L_mp {self.mean_penalty_loss:.4f} \t "
            f"num_s {self.num_steps} eps_len {self.avg_episode_len:.2f}"
        )
