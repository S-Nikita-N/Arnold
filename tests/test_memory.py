"""Characterization tests for arnold.memory (OBCMemory / OBCBatch GAE)."""

import torch
import pytest
import numpy as np

import helpers

from arnold.memory import OBCMemory


########################################
#              OBCMemory               #
########################################

def _fill_memory():
    m = OBCMemory()
    n = 5
    for i in range(n):
        m.states.append(torch.zeros(2, helpers.HISTORY_LEN))
        m.obs_signatures.append([("a", "b")])
        m.action_signatures.append([("c", "d")])
        m.policy_actions.append(torch.zeros(3))
        m.expert_actions.append(torch.zeros(3))
        m.values.append(torch.tensor([float(i) * 0.5]))
        m.log_probs.append(torch.zeros(1))
    m.rewards = [1.0, 0.5, -0.2, 0.8, 1.5]
    m.masks = [1.0, 1.0, 1.0, 1.0, 0.0]
    return m


def test_len_and_clear():
    m = _fill_memory()
    assert len(m) == 5
    m.clear()
    assert len(m) == 0
    assert m.rewards == []


def test_extend():
    a = _fill_memory()
    b = _fill_memory()
    a.extend(b)
    assert len(a) == 10
    assert len(a.rewards) == 10


def test_to_batch_gae_golden(goldens_npz):
    m = _fill_memory()
    batch = m.to_batch(gamma=0.99, tau=0.95)
    assert batch.returns.shape == (5, 1)
    assert batch.advantages.shape == (5, 1)
    np.testing.assert_allclose(
        batch.returns.detach().numpy(),
        goldens_npz["gae_returns"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        batch.advantages.detach().numpy(),
        goldens_npz["gae_advantages"],
        rtol=1e-6,
        atol=1e-6,
    )
    # advantages normalized: near zero mean, ~unit std
    assert batch.advantages.mean().item() == pytest.approx(0.0, abs=1e-6)


def test_to_batch_signatures_and_shapes():
    m = _fill_memory()
    batch = m.to_batch(gamma=0.99, tau=0.95)
    assert batch.obs_signatures == [("a", "b")]
    assert batch.action_signatures == [("c", "d")]
    assert batch.states.shape == (5, 2, helpers.HISTORY_LEN)
    assert batch.rewards.shape == (5,)
    assert batch.values.shape == (5, 1)


def test_to_batch_empty_raises():
    m = OBCMemory()
    with pytest.raises(ValueError, match="empty"):
        m.to_batch(gamma=0.99, tau=0.95)


def test_transfer_dict_roundtrip():
    m = _fill_memory()
    d = m.to_transfer_dict()
    assert d["n"] == 5
    m2 = OBCMemory.from_transfer_dict(d)
    assert len(m2) == 5
    assert m2.rewards == m.rewards
    assert m2.masks == m.masks
    for a, b in zip(m.states, m2.states, strict=False):
        assert torch.equal(a, b)
    # produces identical batch
    b1 = m.to_batch(gamma=0.99, tau=0.95)
    b2 = m2.to_batch(gamma=0.99, tau=0.95)
    assert torch.allclose(b1.returns, b2.returns)


def test_transfer_dict_empty():
    m = OBCMemory()
    d = m.to_transfer_dict()
    assert d == {"n": 0}
    m2 = OBCMemory.from_transfer_dict(d)
    assert len(m2) == 0
