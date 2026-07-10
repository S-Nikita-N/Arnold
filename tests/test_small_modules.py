"""Characterization tests for MLP, SensoryEncoder, dist_utils,
learning_utils.
"""

import torch
import pytest
import numpy as np

import helpers

from arnold import learning_utils
from arnold.torch_model.mlp import MLP
from arnold.torch_model.dist_utils import safe_lrmvn_log_prob
from arnold.torch_model.sensory_encoder import SensoryEncoder


########################################
#                 MLP                  #
########################################

@pytest.mark.parametrize("act", ["tanh", "relu", "sigmoid", "gelu", "silu"])
def test_mlp_activation_dispatch(act):
    torch.manual_seed(0)
    mlp = MLP(4, (8, 2), activation=act)
    out = mlp(torch.randn(3, 4))
    assert out.shape == (3, 2)
    assert mlp.out_dim == 2
    assert len(mlp.affine_layers) == 2


def test_mlp_unknown_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        MLP(4, (8,), activation="banana")


def test_mlp_forward_golden(goldens_npz):
    torch.manual_seed(2)
    mlp = MLP(input_dim=8, hidden_dims=(16, 4), activation="silu")
    torch.manual_seed(101)
    x = torch.randn(3, 8)
    out = mlp(x)
    np.testing.assert_allclose(
        out.detach().numpy(), goldens_npz["mlp_out"], rtol=1e-5, atol=1e-6
    )


########################################
#            SensoryEncoder            #
########################################

def test_sensory_encoder_golden(goldens_npz):
    torch.manual_seed(1)
    enc = SensoryEncoder(
        history_len=helpers.HISTORY_LEN,
        embed_dim=helpers.EMBED_DIM,
    )
    torch.manual_seed(100)
    x = torch.randn(2, 6, helpers.HISTORY_LEN)
    out = enc(x)
    assert out.shape == (2, 6, helpers.EMBED_DIM)
    np.testing.assert_allclose(
        out.detach().numpy(), goldens_npz["sensory_out"], rtol=1e-5, atol=1e-6
    )


########################################
#              dist_utils              #
########################################

def test_safe_lrmvn_log_prob_matches_torch():
    torch.manual_seed(0)
    B, n, k = 4, 6, 3
    loc = torch.randn(B, n)
    cov_factor = torch.randn(B, n, k) * 0.3
    cov_diag = torch.rand(B, n) + 0.5
    dist = torch.distributions.LowRankMultivariateNormal(
        loc, cov_factor, cov_diag,
    )
    value = torch.randn(B, n)
    got = safe_lrmvn_log_prob(dist, value)
    expected = dist.log_prob(value)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, rtol=1e-5, atol=1e-5)


########################################
#            learning_utils            #
########################################

def test_to_test_and_to_train_restore_mode():
    m = torch.nn.Linear(3, 3)
    m.train(True)
    with learning_utils.to_test(m):
        assert m.training is False
    assert m.training is True  # restored

    m.train(False)
    with learning_utils.to_train(m):
        assert m.training is True
    assert m.training is False


def test_to_cpu_and_to_device_restore():
    m = torch.nn.Linear(3, 3)
    orig_device = next(m.parameters()).device
    with learning_utils.to_cpu(m):
        assert next(m.parameters()).device.type == "cpu"
    assert next(m.parameters()).device == orig_device


def test_batch_to_handles_none():
    a = torch.zeros(2)
    out = learning_utils.batch_to("cpu", a, None)
    assert out[0].device.type == "cpu"
    assert out[1] is None


def test_get_optimizer():
    m = torch.nn.Linear(3, 3)
    adam = learning_utils.get_optimizer(
        m, lr=1e-3, weight_decay=0.0, optimizer_type="adam",
    )
    assert isinstance(adam, torch.optim.Adam)
    sgd = learning_utils.get_optimizer(
        m, lr=1e-3, weight_decay=0.0, optimizer_type="sgd",
    )
    assert isinstance(sgd, torch.optim.SGD)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        learning_utils.get_optimizer(
            m, lr=1e-3, weight_decay=0.0, optimizer_type="rmsprop",
        )
