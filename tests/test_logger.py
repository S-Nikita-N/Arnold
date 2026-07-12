"""Characterization tests for arnold.logger.OBCLogger.

Note: per the task constraints we always call reset_losses()/set_update_losses()
before get_log_str, and do not pin total_loss composition wrt load_balance /
mean_penalty omission.
"""

import math

from arnold.logger import OBCLogger


########################################
#              OBCLogger               #
########################################


def _make_logged():
    lg = OBCLogger()
    lg.step(1.0, info={"comp_a": 0.5, "done": True})
    lg.step(2.0, info={"comp_a": 1.5, "done": False})
    lg.end_episode()
    lg.step(3.0, info={"comp_a": 2.0})
    lg.end_episode()
    return lg


def test_step_and_episode_tracking():
    lg = _make_logged()
    assert lg.num_steps == 3
    assert lg.num_episodes == 2
    assert lg.total_reward == 6.0
    assert lg.episode_rewards == [3.0, 3.0]
    assert lg.episode_lengths == [2, 1]
    assert lg.min_reward == 1.0
    assert lg.max_reward == 3.0
    assert lg.min_episode_reward == 3.0
    assert lg.max_episode_reward == 3.0


def test_info_dict_bool_and_numeric():
    lg = _make_logged()
    # bool converted to 1.0/0.0
    assert lg.info_dict["done"] == [1.0, 0.0]
    assert lg.info_dict["comp_a"] == [0.5, 1.5, 2.0]


def test_properties():
    lg = _make_logged()
    assert lg.avg_episode_reward == 3.0  # 6/2
    assert lg.avg_episode_len == 1.5  # 3/2
    assert lg.avg_reward == 2.0  # 6/3


def test_properties_empty():
    lg = OBCLogger()
    assert lg.avg_episode_reward == 0.0
    assert lg.avg_episode_len == 0.0
    assert lg.avg_reward == 0.0


def test_total_loss_sum():
    lg = OBCLogger()
    lg.reset_losses()
    lg.set_update_losses(
        ppo_loss=1.0,
        imitation_loss=2.0,
        value_loss=3.0,
        entropy_loss=4.0,
        load_balance_loss=5.0,
        mean_penalty_loss=6.0,
    )
    # total_loss sums all six loss components (ppo+im+val+ent+bal+mp)
    assert lg.total_loss == 21.0


def test_fresh_logger_has_all_loss_attrs():
    # load_balance_loss must be initialized in __init__ so get_log_str /
    # total_loss work before any set_update_losses/reset_losses call.
    lg = OBCLogger()
    assert lg.load_balance_loss == 0.0
    assert lg.mean_penalty_loss == 0.0
    assert lg.total_loss == 0.0
    # get_log_str must not raise on a freshly constructed logger
    s = lg.get_log_str(epoch=0, expert_name="legs")
    assert "L_bal 0.0000" in s


def test_reset_losses():
    lg = OBCLogger()
    lg.set_update_losses(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    lg.reset_losses()
    assert lg.ppo_loss == 0.0
    assert lg.load_balance_loss == 0.0
    assert lg.mean_penalty_loss == 0.0
    assert lg.total_loss == 0.0


def test_reset_episodes():
    lg = _make_logged()
    lg.reset_episodes()
    assert lg.episode_rewards == []
    assert lg.episode_lengths == []
    assert len(lg.info_dict) == 0


def test_merge():
    a = _make_logged()
    b = _make_logged()
    merged = OBCLogger.merge([a, None, b])
    assert merged.num_steps == 6
    assert merged.num_episodes == 4
    assert merged.total_reward == 12.0
    assert merged.episode_rewards == [3.0, 3.0, 3.0, 3.0]
    assert merged.info_dict["comp_a"] == [0.5, 1.5, 2.0, 0.5, 1.5, 2.0]


def test_merge_empty_list():
    merged = OBCLogger.merge([])
    assert merged.num_steps == 0
    assert merged.min_reward == math.inf
    assert merged.max_reward == -math.inf


def test_get_log_str_after_reset_losses():
    lg = _make_logged()
    lg.reset_losses()
    lg.set_update_losses(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    lg.sample_time = 1.23
    lg.update_time = 4.56
    s = lg.get_log_str(epoch=7, expert_name="legs")
    assert "Ep: 7" in s
    assert "legs" in s
    assert "L_ppo 0.1000" in s
    assert "L_bal 0.5000" in s
    assert "L_mp 0.6000" in s
    assert "num_s 3" in s
