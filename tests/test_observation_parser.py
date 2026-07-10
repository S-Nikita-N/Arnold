"""Characterization tests for arnold.observation_parser."""

import torch
import pytest
import numpy as np

import helpers

from arnold.observation_parser import BodyGroup, ObservationParser


########################################
#          ObservationParser           #
########################################

def test_parse_side():
    op = helpers.make_observation_parser()
    assert op._parse_side("femur_r") == ("femur", "r")
    assert op._parse_side("femur_l") == ("femur", "l")
    assert op._parse_side("pelvis") == ("pelvis", "c")


def test_observation_specs_golden(goldens):
    op = helpers.make_observation_parser()
    assert len(op.obs_specs) == goldens["obs_n_specs"]
    assert op.n_obs_elements == goldens["obs_n_elements"]
    assert len(op.obs_signatures) == goldens["obs_n_signatures"]
    names_sizes = [[s.name, s.size] for s in op.obs_specs]
    assert names_sizes == goldens["obs_spec_names_sizes"]
    # size == number of signatures per spec
    for spec in op.obs_specs:
        assert len(spec.signatures) == spec.size


def test_obs_signatures_flat_matches_specs():
    op = helpers.make_observation_parser()
    flat = []
    for spec in op.obs_specs:
        flat.extend(spec.signatures)
    assert op.obs_signatures == flat


def test_specific_signatures():
    op = helpers.make_observation_parser()
    sigs = op.obs_signatures
    # first spec is root_height
    assert sigs[0] == ("root", "c", "height")
    # root_tilt next (4)
    assert sigs[1] == ("root", "c", "tilt", "x")
    assert sigs[4] == ("root", "c", "tilt", "qy")


@pytest.mark.parametrize(("granularity", "key"), [
    ("scalar", "obs_groups_scalar"),
    ("per_spec", "obs_groups_per_spec"),
    ("per_body", "obs_groups_per_body"),
])
def test_body_groups_counts(granularity, key, goldens):
    op = helpers.make_observation_parser()
    groups = op.get_body_groups(granularity)
    assert len(groups) == goldens[key]
    assert all(isinstance(g, BodyGroup) for g in groups)


def test_body_groups_scalar_indices_cover_all():
    op = helpers.make_observation_parser()
    groups = op.get_body_groups("scalar")
    # each scalar group has exactly one index; union covers all elements once
    all_idx = []
    for g in groups:
        assert len(g.indices) == 1
        all_idx.extend(g.indices)
    assert sorted(all_idx) == list(range(op.n_obs_elements))


def test_body_groups_per_body_indices_partition():
    op = helpers.make_observation_parser()
    groups = op.get_body_groups("per_body")
    all_idx = []
    for g in groups:
        all_idx.extend(g.indices)
    # per_body must be a partition of all obs indices
    assert sorted(all_idx) == list(range(op.n_obs_elements))


def test_body_groups_unknown_raises():
    op = helpers.make_observation_parser()
    with pytest.raises(ValueError, match="Unknown granularity"):
        op.get_body_groups("nonsense")


def test_reset_update_get_observation():
    op = helpers.make_observation_parser()
    n = op.n_obs_elements
    obs0 = np.arange(n, dtype=np.float64)
    op.reset(obs0)
    # history filled with the same obs across time
    assert op.history.shape == (n, op.history_len)
    assert np.allclose(op.history[:, 0], obs0)
    assert np.allclose(op.history[:, -1], obs0)

    obs1 = np.arange(n, dtype=np.float64) + 100.0
    op.update(obs1)
    # newest column is obs1, rolled
    assert np.allclose(op.history[:, -1], obs1)
    assert np.allclose(op.history[:, -2], obs0)

    ts, sigs = op.get_observation()
    assert ts.shape == (1, n, op.history_len)
    assert ts.dtype == torch.float32
    assert sigs is op.obs_signatures


def test_update_before_reset_raises():
    op = helpers.make_observation_parser()
    with pytest.raises(RuntimeError, match="Call reset"):
        op.update(np.zeros(op.n_obs_elements))


def test_get_observation_before_reset_raises():
    op = helpers.make_observation_parser()
    with pytest.raises(RuntimeError, match="Call reset"):
        op.get_observation()


def test_partial_inputs_change_specs():
    """Fewer proprioceptive inputs => fewer specs (structural
    characterization).
    """
    op = ObservationParser(
        body_names=helpers.BODY_NAMES,
        muscle_names=helpers.MUSCLE_NAMES,
        proprioceptive_inputs=["root_height", "muscle_len"],
        task_inputs=[],
        track_bodies=helpers.TRACK_BODIES,
        num_bodies=len(helpers.BODY_NAMES),
        num_muscles=len(helpers.MUSCLE_NAMES),
        history_len=helpers.HISTORY_LEN,
    )
    assert [s.name for s in op.obs_specs] == ["root_height", "muscle_len"]
    assert op.n_obs_elements == 1 + len(helpers.MUSCLE_NAMES)
