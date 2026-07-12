"""Characterization tests for the policy forward passes.

Pins the deterministic mean / cov_factor / diag_std / value / latent_std
outputs of every policy (LatticePolicy, MoELatticePolicy, TransformerPolicy,
GateMoEPolicy) plus the action-distribution / entropy / log-prob helpers.
All CPU-only, seeded, tiny synthetic inputs — no env, wandb, or submodule.

Torch float32 matmul reduction order differs across BLAS/arch, so golden
comparisons use a cross-platform tolerance (rtol=1e-4, atol=1e-5).
"""

import torch
import pytest
import numpy as np

import helpers as H

RTOL = 1e-4
ATOL = 1e-5


########################################
#           Forward goldens            #
########################################


def _check(prefix, out, goldens, goldens_npz):
    names = ["mean", "cov", "diag_std", "value", "latent_std", "lb"]
    for name, tensor in zip(names, out, strict=False):
        key = f"{prefix}_{name}"
        if tensor is None:
            assert goldens.get(key) is None
            continue
        assert list(tensor.shape) == goldens[f"{key}_shape"]
        np.testing.assert_allclose(
            tensor.detach().numpy(),
            goldens_npz[key],
            rtol=RTOL,
            atol=ATOL,
        )


def test_lattice_forward_golden(goldens, goldens_npz):
    op, ap, obs = H.make_policy_io()
    net = H.build_lattice(op.n_obs_elements, len(ap.action_signatures))
    out = net(obs, op.obs_signatures, ap.action_signatures, "legs")
    # (mean, cov_factor, diag_std, value, latent_std)
    assert out[0].shape == (H.POLICY_BATCH, len(ap.action_signatures))
    _check("lat", out, goldens, goldens_npz)


def test_moe_forward_golden(goldens, goldens_npz):
    op, ap, obs = H.make_policy_io()
    net = H.build_moe(op.n_obs_elements, len(ap.action_signatures))
    out = net(obs, op.obs_signatures, ap.action_signatures, "legs")
    # (mean, cov, diag_std, value, latent_std, load_balance_loss)
    assert len(out) == 6
    assert out[5].shape == (1,)  # load-balance loss scalar
    _check("moe_fwd", out, goldens, goldens_npz)


def test_transformer_forward_golden(goldens, goldens_npz):
    op, ap, obs = H.make_policy_io()
    net = H.build_transformer(op)
    out = net(obs, op.obs_signatures, ap.action_signatures, "legs")
    # transformer cov_factor is per-sample: [B, A, L]
    assert out[1].shape == (H.POLICY_BATCH, len(ap.action_signatures), 4)
    _check("tp", out, goldens, goldens_npz)


def test_transformer_granulated_forward_golden(goldens, goldens_npz):
    # action_granulation="hybrid" exercises the hierarchical action decoder.
    op, ap, obs = H.make_policy_io()
    net = H.build_transformer_granulated(op, ap)
    out = net(obs, op.obs_signatures, ap.action_signatures, "legs")
    assert out[0].shape == (H.POLICY_BATCH, len(ap.action_signatures))
    _check("tpg", out, goldens, goldens_npz)


def test_gate_moe_forward_golden(goldens, goldens_npz, tmp_path):
    op, ap, obs = H.make_policy_io()
    net = H.build_gate_moe(
        op.n_obs_elements,
        len(ap.action_signatures),
        tmp_path,
    )
    out = net(obs, op.obs_signatures, ap.action_signatures, "legs")
    # (gate_logits[B, n_experts], None, None, value[B, 1], None)
    assert out[0].shape == (H.POLICY_BATCH, 2)
    assert out[1] is None
    assert out[2] is None
    assert out[4] is None
    _check("gm", out, goldens, goldens_npz)


########################################
#           Distribution API           #
########################################


def test_lattice_action_dist_and_helpers():
    op, ap, obs = H.make_policy_io()
    net = H.build_lattice(op.n_obs_elements, len(ap.action_signatures))
    mean, cov, diag_std, _, _ = net(
        obs,
        op.obs_signatures,
        ap.action_signatures,
        "legs",
    )
    dist = net.build_action_dist(mean, cov, diag_std)
    # entropy is a finite scalar; log_prob has one row per batch element
    ent = net.dist_entropy(dist)
    assert ent.shape == ()
    assert torch.isfinite(ent)
    actions = mean.detach()
    lp = net.dist_log_prob(dist, actions)
    assert lp.shape == (H.POLICY_BATCH, 1)
    assert torch.isfinite(lp).all()


def test_moe_forward_deterministic():
    # No random tie-breaker anymore → repeated eval forwards are identical.
    op, ap, obs = H.make_policy_io()
    net = H.build_moe(op.n_obs_elements, len(ap.action_signatures))
    o1 = net(obs, op.obs_signatures, ap.action_signatures, "legs")
    o2 = net(obs, op.obs_signatures, ap.action_signatures, "legs")
    assert torch.allclose(o1[0], o2[0])
    assert torch.allclose(o1[3], o2[3])


@pytest.mark.parametrize("builder", ["transformer", "moe"])
def test_get_action_and_dist_helpers(builder):
    op, ap, obs = H.make_policy_io()
    if builder == "transformer":
        net = H.build_transformer(op)
    else:
        net = H.build_moe(op.n_obs_elements, len(ap.action_signatures))

    # get_action → (env_action, store_action, log_prob, value)
    torch.manual_seed(0)
    env_action, store_action, log_prob, value = net.get_action(
        obs,
        op.obs_signatures,
        ap.action_signatures,
        "legs",
        deterministic=False,
    )
    A = len(ap.action_signatures)
    assert env_action.shape == (H.POLICY_BATCH, A)
    assert torch.isfinite(env_action).all()
    assert torch.equal(env_action, store_action)
    assert log_prob.shape == (H.POLICY_BATCH, 1)
    assert value.shape == (H.POLICY_BATCH, 1)

    # deterministic path returns the mean (no sampling)
    det_action, _, _, _ = net.get_action(
        obs,
        op.obs_signatures,
        ap.action_signatures,
        "legs",
        deterministic=True,
    )
    assert det_action.shape == (H.POLICY_BATCH, A)

    # build_action_dist → entropy scalar + per-row log_prob
    fwd = net(obs, op.obs_signatures, ap.action_signatures, "legs")
    mean, cov, diag_std = fwd[0], fwd[1], fwd[2]
    dist = net.build_action_dist(mean, cov, diag_std)
    ent = net.dist_entropy(dist)
    assert torch.isfinite(ent).all()
    lp = net.dist_log_prob(dist, mean.detach())
    assert lp.shape[0] == H.POLICY_BATCH
    assert torch.isfinite(lp).all()


def test_gate_moe_empty_checkpoints_raises():
    from myotrainer.torch_model.policy_gate_moe import GateMoEPolicy

    with pytest.raises(ValueError, match="expert_checkpoints"):
        GateMoEPolicy(state_dim=8, action_dim=4, expert_checkpoints=[])


def test_policies_return_value_toggle():
    op, ap, obs = H.make_policy_io()
    net = H.build_lattice(op.n_obs_elements, len(ap.action_signatures))
    out = net(
        obs,
        op.obs_signatures,
        ap.action_signatures,
        "legs",
        return_value=False,
    )
    assert out[3] is None  # value suppressed
