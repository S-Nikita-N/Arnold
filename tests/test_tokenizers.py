"""Characterization tests for BodyTokenizer and ActionTokenizer."""

import torch
import pytest
import numpy as np

import helpers

from arnold.torch_model.body_tokenizer import BodyTokenizer
from arnold.torch_model.action_tokenizer import ActionTokenizer


########################################
#            BodyTokenizer             #
########################################


def _build_body_tokenizer(op):
    groups = op.get_body_groups("per_body")
    torch.manual_seed(3)
    return BodyTokenizer(
        groups={"legs": groups},
        history_len=helpers.HISTORY_LEN,
        embed_dim=helpers.EMBED_DIM,
    )


def test_body_tokenizer_structure(goldens):
    op = helpers.make_observation_parser()
    tok = _build_body_tokenizer(op)
    assert tok.expert_names == ["legs"]
    assert tok._expert_type_orders["legs"] == goldens["body_tok_types"]
    assert len(tok.get_group_signatures("legs")) == goldens["body_tok_n_groups"]
    # one projector per type
    assert sorted(tok.projectors.keys()) == goldens["body_tok_types"]


def test_body_tokenizer_encode_golden(goldens, goldens_npz):
    op = helpers.make_observation_parser()
    tok = _build_body_tokenizer(op)
    torch.manual_seed(102)
    obs_ts = torch.randn(2, op.n_obs_elements, helpers.HISTORY_LEN)
    out = tok.encode(obs_ts, "legs")
    assert list(out.shape) == goldens["body_tok_out_shape"]
    np.testing.assert_allclose(
        out.detach().numpy(),
        goldens_npz["body_tok_out"],
        rtol=1e-4,
        atol=1e-5,
    )


def test_body_tokenizer_feature_mismatch_raises():
    """Two experts with differing feature counts for the same type
    -> ValueError.
    """
    from arnold.observation_parser import BodyGroup

    g_a = [
        BodyGroup(
            name="m1",
            side="c",
            type="muscle",
            indices=[0, 1],
            signature=("m1", "c", "muscle"),
        ),
    ]
    g_b = [
        BodyGroup(
            name="m2",
            side="c",
            type="muscle",
            indices=[0, 1, 2],
            signature=("m2", "c", "muscle"),
        ),
    ]
    with pytest.raises(ValueError, match="Feature count mismatch"):
        BodyTokenizer(
            groups={"a": g_a, "b": g_b},
            history_len=helpers.HISTORY_LEN,
            embed_dim=helpers.EMBED_DIM,
        )


########################################
#           ActionTokenizer            #
########################################


def _build_action_tokenizer(ap):
    grouping = ap.get_muscle_grouping("hybrid")
    torch.manual_seed(4)
    tok = ActionTokenizer(
        groupings={"legs": grouping},
        action_signatures={"legs": ap.action_signatures},
        embed_dim=helpers.EMBED_DIM,
        strategy="hybrid",
    )
    return tok, grouping


def test_action_tokenizer_structure(goldens):
    ap = helpers.make_action_parser()
    tok, grouping = _build_action_tokenizer(ap)
    assert grouping.n_groups == goldens["atok_n_groups"]
    assert tok.expert_group_sizes["legs"] == goldens["atok_group_sizes"]
    got_sigs = [list(s) for s in tok.get_group_signatures("legs")]
    assert got_sigs == goldens["atok_group_signatures"]
    got_msigs = [list(s) for s in tok.get_muscle_signatures("legs")]
    assert got_msigs == goldens["atok_muscle_sigs"]


def test_action_tokenizer_build_select_idx():
    ap = helpers.make_action_parser()
    tok, _ = _build_action_tokenizer(ap)
    # pectoralis canonical = [PECM1, PECM2, PECM3]; expert has all three sided
    idx = tok.build_select_idx("pectoralis", ["PECM1_r", "PECM2_r", "PECM3_r"])
    assert idx == [0, 1, 2]
    # subset / reorder
    idx2 = tok.build_select_idx("pectoralis", ["PECM3_r", "PECM1_r"])
    assert idx2 == [2, 0]
    # unknown granule base -> identity range
    idx3 = tok.build_select_idx("not_canonical", ["x_r", "y_r"])
    assert idx3 == [0, 1]


def test_action_tokenizer_reorder_idx_inverts_grouping():
    ap = helpers.make_action_parser()
    tok, grouping = _build_action_tokenizer(ap)
    reorder = tok.reorder_idx_legs.tolist()
    # grouped order sequence of muscles
    grouped_muscles = []
    for gid in grouping.group_order:
        grouped_muscles.extend(grouping.groups[gid])
    # applying reorder to grouped order must recover original muscle order
    recovered = [grouped_muscles[i] for i in reorder]
    assert recovered == helpers.MUSCLE_NAMES


def test_action_tokenizer_decode_golden(goldens, goldens_npz):
    ap = helpers.make_action_parser()
    tok, grouping = _build_action_tokenizer(ap)
    torch.manual_seed(103)
    group_out = torch.randn(2, grouping.n_groups, helpers.EMBED_DIM)
    mean, hidden = tok.decode(group_out, "legs")
    assert list(mean.shape) == goldens["atok_mean_shape"]
    assert list(hidden.shape) == goldens["atok_hidden_shape"]
    np.testing.assert_allclose(
        mean.detach().numpy(),
        goldens_npz["atok_mean"],
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        hidden.detach().numpy(),
        goldens_npz["atok_hidden"],
        rtol=1e-4,
        atol=1e-5,
    )
