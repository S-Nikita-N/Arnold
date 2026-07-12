"""Stage-2 tests: require a real GPU/MPS accelerator (or RUN_STAGE2=1).

These are skipped by default (stage-1 is CPU-only). They assert the policies
are device-agnostic: the same seeded network produces the same forward output
on the accelerator as on CPU, within float tolerance.
"""

import torch
import pytest

import helpers as H

pytestmark = pytest.mark.stage2


########################################
#            Device parity             #
########################################


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    pytest.skip("no GPU/MPS accelerator available")


def test_lattice_forward_matches_cpu():
    op, ap, obs = H.make_policy_io()
    state_dim, action_dim = op.n_obs_elements, len(ap.action_signatures)

    cpu_net = H.build_lattice(state_dim, action_dim)
    cpu_mean = cpu_net(
        obs,
        op.obs_signatures,
        ap.action_signatures,
        "legs",
    )[0]

    dev = _device()
    dev_net = H.build_lattice(state_dim, action_dim).to(dev)
    dev_mean = dev_net(
        obs.to(dev),
        op.obs_signatures,
        ap.action_signatures,
        "legs",
    )[0]

    assert dev_mean.device.type == dev.type
    torch.testing.assert_close(
        dev_mean.cpu(),
        cpu_mean,
        rtol=1e-4,
        atol=1e-5,
    )


def test_transformer_forward_matches_cpu():
    op, ap, obs = H.make_policy_io()

    cpu_mean = H.build_transformer(op)(
        obs,
        op.obs_signatures,
        ap.action_signatures,
        "legs",
    )[0]

    dev = _device()
    dev_mean = H.build_transformer(op).to(dev)(
        obs.to(dev),
        op.obs_signatures,
        ap.action_signatures,
        "legs",
    )[0]

    assert dev_mean.device.type == dev.type
    torch.testing.assert_close(
        dev_mean.cpu(),
        cpu_mean,
        rtol=1e-3,
        atol=1e-4,
    )
