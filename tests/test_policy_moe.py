"""Characterization tests for arnold.torch_model.policy_moe.

We only pin the three DETERMINISTIC helper methods
(compute_sparse_balance_loss, compute_soft_balance_loss, _compute_cov_factor)
plus shape/invariant checks on gate_forward.  We do NOT assert exact MoE
routing outputs (gate_forward uses a random tie-breaker).
"""

import torch
import pytest
import numpy as np

from arnold.torch_model.policy_moe import MoELatticePolicy


ACTION_DIM = 6
LATENT_DIM = 4


########################################
#              MoE policy              #
########################################

def _make_sparse(seed=5, cov_from_experts=False):
    torch.manual_seed(seed)
    return MoELatticePolicy(
        state_dim=8, action_dim=ACTION_DIM,
        shared_units=(), shared_expert_units=(LATENT_DIM,),
        num_experts=4, top_k=2, expert_units=(LATENT_DIM,),
        gate_units=(8,), value_units=(8,),
        cov_from_experts=cov_from_experts,
    )


def _make_soft(seed=6):
    torch.manual_seed(seed)
    return MoELatticePolicy(
        state_dim=8, action_dim=ACTION_DIM,
        shared_units=(), shared_expert_units=(LATENT_DIM,),
        num_experts=4, top_k=4, expert_units=(LATENT_DIM,),
        gate_units=(8,), value_units=(8,),
    )


def _fixed_inputs():
    top_k_indices = torch.tensor([[0, 1], [2, 3], [0, 2]], dtype=torch.long)
    torch.manual_seed(106)
    gate_probs = torch.softmax(torch.randn(3, 4), dim=-1)
    return top_k_indices, gate_probs


def test_config_flags():
    sparse = _make_sparse()
    assert sparse.soft_routing is False   # top_k != num_experts
    assert sparse.latent_dim == LATENT_DIM
    soft = _make_soft()
    assert soft.soft_routing is True      # top_k == num_experts


def test_shared_expert_units_mismatch_raises():
    with pytest.raises(ValueError, match="latent_dim"):
        MoELatticePolicy(
            state_dim=8, action_dim=ACTION_DIM,
            shared_units=(), shared_expert_units=(4,),
            num_experts=2, top_k=1, expert_units=(8,),   # 8 != 4
            gate_units=(8,), value_units=(8,),
        )


def test_compute_sparse_balance_loss_golden(goldens):
    sparse = _make_sparse()
    top_k_indices, gate_probs = _fixed_inputs()
    loss, stats = sparse.compute_sparse_balance_loss(top_k_indices, gate_probs)
    assert loss.shape == (1,)
    assert float(loss.item()) == pytest.approx(
        goldens["moe_sparse_lb"], rel=1e-5,
    )
    assert stats["lb_total"] == pytest.approx(
        goldens["moe_sparse_lb"], rel=1e-5,
    )


def test_compute_soft_balance_loss_golden(goldens):
    soft = _make_soft()
    _, gate_probs = _fixed_inputs()
    loss, stats = soft.compute_soft_balance_loss(gate_probs)
    assert loss.shape == (1,)
    assert float(loss.item()) == pytest.approx(goldens["moe_soft_lb"], rel=1e-5)
    for k, v in goldens["moe_soft_stats"].items():
        assert stats[k] == pytest.approx(v, rel=1e-5)


def test_compute_cov_factor_shared_only_golden(goldens, goldens_npz):
    sparse = _make_sparse()  # cov_from_experts=False
    top_k_indices, _ = _fixed_inputs()
    top_k_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]])
    cov_f, diag_std, latent_std = sparse._compute_cov_factor(
        top_k_indices, top_k_weights,
    )
    # shared-only path: cov_factor is [A, L] (no batch dim)
    assert list(cov_f.shape) == goldens["moe_cov_shared_shape"]
    assert diag_std.shape == (1, ACTION_DIM)
    assert latent_std.shape == (LATENT_DIM,)
    np.testing.assert_allclose(
        cov_f.detach().numpy(),
        goldens_npz["moe_cov_shared"],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        diag_std.detach().numpy(),
        goldens_npz["moe_diag_std"],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        latent_std.detach().numpy(),
        goldens_npz["moe_latent_std"],
        rtol=1e-5,
        atol=1e-6,
    )
    # std must be strictly positive (softplus + min_diag_std)
    assert (diag_std > 0).all()
    assert (latent_std > 0).all()


def test_compute_cov_factor_per_sample_golden(goldens, goldens_npz):
    per = _make_sparse(seed=7, cov_from_experts=True)
    top_k_indices, _ = _fixed_inputs()
    top_k_weights = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.5, 0.5]])
    cov_f, _, _ = per._compute_cov_factor(top_k_indices, top_k_weights)
    # per-sample path: [B, A, L]
    assert list(cov_f.shape) == goldens["moe_cov_persample_shape"]
    np.testing.assert_allclose(
        cov_f.detach().numpy(),
        goldens_npz["moe_cov_persample"],
        rtol=1e-5,
        atol=1e-6,
    )


def test_gate_forward_shapes_and_invariants():
    sparse = _make_sparse()
    torch.manual_seed(42)
    h = torch.randn(5, 8)
    top_k_weights, top_k_indices, gate_probs = sparse.gate_forward(h)
    assert top_k_weights.shape == (5, sparse.top_k)
    assert top_k_indices.shape == (5, sparse.top_k)
    assert gate_probs.shape == (5, sparse.num_experts)
    # gate_probs is a valid softmax distribution
    assert torch.allclose(gate_probs.sum(dim=-1), torch.ones(5), atol=1e-5)
    # top_k weights normalized to sum ~1 per row
    assert torch.allclose(top_k_weights.sum(dim=-1), torch.ones(5), atol=1e-4)
    # indices are valid experts
    assert top_k_indices.min() >= 0
    assert top_k_indices.max() < sparse.num_experts


def test_gate_forward_is_deterministic():
    # After removing the random tie-breaker, repeated calls on the same input
    # (in eval) must return identical routing.
    sparse = _make_sparse()
    sparse.eval()
    h = torch.randn(5, 8)
    w1, i1, p1 = sparse.gate_forward(h)
    w2, i2, p2 = sparse.gate_forward(h)
    assert torch.equal(i1, i2)
    assert torch.allclose(w1, w2)
    assert torch.allclose(p1, p2)
