"""Characterization tests for arnold.torch_model.normalization."""

import torch
import pytest
import numpy as np

import helpers

from arnold.torch_model.normalization import SignatureNormalizerModule

SIGS = [("a", "b"), ("c", "d"), ("e", "f")]


########################################
#            Normalization             #
########################################


def _build_and_fill():
    norm = SignatureNormalizerModule()
    torch.manual_seed(104)
    vals1 = torch.randn(4, 3, helpers.HISTORY_LEN)
    vals2 = torch.randn(4, 3, helpers.HISTORY_LEN)
    norm.update(SIGS, vals1)
    norm.update(SIGS, vals2)
    return norm


def test_sig_key():
    assert SignatureNormalizerModule._sig_key(("a", "b", "c")) == "a|b|c"
    assert SignatureNormalizerModule._sig_key(["x", "y"]) == "x|y"
    assert SignatureNormalizerModule._sig_key("z") == "z"


def test_welford_stats_golden(goldens):
    norm = _build_and_fill()
    g = goldens["norm_stats"]
    assert set(norm.stats.keys()) == set(g.keys())
    for key, (count, mean_t, m2_t) in norm.stats.items():
        assert count == g[key]["count"]
        assert float(mean_t) == pytest.approx(g[key]["mean"], rel=1e-6)
        assert float(m2_t) == pytest.approx(g[key]["M2"], rel=1e-6)


def test_normalize_golden(goldens_npz):
    norm = _build_and_fill()
    test_vals = torch.arange(
        3 * helpers.HISTORY_LEN,
        dtype=torch.float32,
    ).reshape(1, 3, helpers.HISTORY_LEN)
    out = norm.normalize(SIGS, test_vals)
    assert out.shape == test_vals.shape
    np.testing.assert_allclose(
        out.detach().numpy(),
        goldens_npz["norm_normalized"],
        rtol=1e-6,
        atol=1e-6,
    )


def test_normalize_unknown_signatures_passthrough():
    """With no stats (or count<2) values are returned unchanged (cloned)."""
    norm = SignatureNormalizerModule()
    vals = torch.randn(2, 3, helpers.HISTORY_LEN)
    out = norm.normalize(SIGS, vals)
    assert torch.equal(out, vals)
    assert out is not vals  # clone


def test_normalize_matches_manual_formula():
    norm = _build_and_fill()
    vals = torch.arange(
        3 * helpers.HISTORY_LEN,
        dtype=torch.float32,
    ).reshape(1, 3, helpers.HISTORY_LEN)
    out = norm.normalize(SIGS, vals)
    keys = [SignatureNormalizerModule._sig_key(s) for s in SIGS]
    for i, key in enumerate(keys):
        count, mean, m2 = norm.stats[key]
        std = torch.sqrt(m2 / count + 1e-8)
        expected = (vals[:, i, :] - mean) / std
        assert torch.allclose(out[:, i, :], expected, rtol=1e-6, atol=1e-6)


def test_get_index_map_returns_three_tuple():
    # documented + actual return is (known_t, unknown_t, keys)
    norm = _build_and_fill()
    result = norm.get_index_map(SIGS, torch.device("cpu"))
    assert len(result) == 3
    known_t, unknown_t, keys = result
    assert known_t.dtype == torch.long
    assert unknown_t.dtype == torch.long
    assert keys == [SignatureNormalizerModule._sig_key(s) for s in SIGS]
    # all three sigs have count>=2 after two updates → all known, none unknown
    assert known_t.tolist() == [0, 1, 2]
    assert unknown_t.numel() == 0


def test_get_index_map_caches_same_object():
    norm = _build_and_fill()
    dev = torch.device("cpu")
    r1 = norm.get_index_map(SIGS, dev)
    r2 = norm.get_index_map(SIGS, dev)
    # second call on same device returns the cached tuple unchanged
    assert r1[0] is r2[0]
    assert r1[1] is r2[1]


def test_extra_state_roundtrip():
    norm = _build_and_fill()
    state = norm.get_extra_state()
    norm2 = SignatureNormalizerModule()
    norm2.set_extra_state(state)
    assert set(norm2.stats.keys()) == set(norm.stats.keys())
    for key in norm.stats:
        c1, m1, v1 = norm.stats[key]
        c2, m2, v2 = norm2.stats[key]
        assert c1 == c2
        assert torch.allclose(torch.as_tensor(m1), torch.as_tensor(m2))
        assert torch.allclose(torch.as_tensor(v1), torch.as_tensor(v2))
    # normalized output identical after roundtrip
    vals = torch.arange(
        3 * helpers.HISTORY_LEN,
        dtype=torch.float32,
    ).reshape(1, 3, helpers.HISTORY_LEN)
    assert torch.allclose(
        norm.normalize(SIGS, vals),
        norm2.normalize(SIGS, vals),
    )
