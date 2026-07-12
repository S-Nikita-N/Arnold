"""Characterization tests for arnold.action_parser (pure, no torch/env)."""

import pytest

import helpers

from arnold.action_parser import (
    ActionParser,
    MuscleGrouping,
    HYBRID_LIMB_MAP,
    FUNCTIONAL_LIMB_MAP,
    get_canonical_granules,
    CANONICAL_GRANULES_HYBRID,
    ANATOMICAL_TORSO_REGEX_MAP,
    FUNCTIONAL_TORSO_REGEX_MAP,
    CANONICAL_GRANULES_FUNCTIONAL,
    CANONICAL_GRANULES_ANATOMICAL,
)


########################################
#            Static helpers            #
########################################


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("PECM1_r", ("PECM1", "r")),
        ("DELT2_l", ("DELT2", "l")),
        ("psoas", ("psoas", "c")),
        ("rect_abd", ("rect_abd", "c")),
        ("", ("", "c")),
    ],
)
def test_parse_side(name, expected):
    assert ActionParser.parse_side(name) == expected


@pytest.mark.parametrize(
    ("gid", "base"),
    [
        ("pectoralis_r", "pectoralis"),
        ("hip_flexors_l", "hip_flexors"),
        ("ercspn", "ercspn"),  # no side suffix
        ("psoas_c", "psoas_c"),  # _c is not stripped (only _r/_l)
    ],
)
def test_granule_base_from_group_id(gid, base):
    assert MuscleGrouping.granule_base_from_group_id(gid) == base


@pytest.mark.parametrize(
    ("gid", "side"),
    [
        ("pectoralis_r", "r"),
        ("hip_flexors_l", "l"),
        ("ercspn", "c"),
    ],
)
def test_parse_side_from_group_id(gid, side):
    assert MuscleGrouping.parse_side_from_group_id(gid) == side


########################################
#          Action signatures           #
########################################


def test_build_action_signatures():
    ap = helpers.make_action_parser()
    assert len(ap.action_signatures) == len(helpers.MUSCLE_NAMES)
    assert ap.num_muscles == len(helpers.MUSCLE_NAMES)
    # every signature: (base, side, "muscle", "activation")
    assert ap.action_signatures[0] == ("PECM1", "r", "muscle", "activation")
    for sig, muscle in zip(
        ap.action_signatures,
        helpers.MUSCLE_NAMES,
        strict=False,
    ):
        base, side = ActionParser.parse_side(muscle)
        assert sig == (base, side, "muscle", "activation")


########################################
#             assign_group             #
########################################


def test_assign_group_torso_regex():
    ap = helpers.make_action_parser()
    # regex match takes side suffix from the muscle
    assert (
        ap.assign_group(
            "PECM1_r",
            ANATOMICAL_TORSO_REGEX_MAP,
            HYBRID_LIMB_MAP,
        )
        == "pectoralis_r"
    )
    assert (
        ap.assign_group(
            "DELT2_l",
            FUNCTIONAL_TORSO_REGEX_MAP,
            FUNCTIONAL_LIMB_MAP,
        )
        == "shoulder_abductors_l"
    )


def test_assign_group_limb_dict():
    ap = helpers.make_action_parser()
    assert (
        ap.assign_group(
            "psoas_r",
            ANATOMICAL_TORSO_REGEX_MAP,
            HYBRID_LIMB_MAP,
        )
        == "hip_flexors_r"
    )
    # center muscle -> no side suffix
    assert (
        ap.assign_group(
            "psoas",
            ANATOMICAL_TORSO_REGEX_MAP,
            HYBRID_LIMB_MAP,
        )
        == "hip_flexors"
    )


def test_assign_group_singleton_fallback():
    ap = helpers.make_action_parser()
    assert (
        ap.assign_group(
            "unknownmuscle_r",
            ANATOMICAL_TORSO_REGEX_MAP,
            {},
            singleton_fallback=True,
        )
        == "unknownmuscle_r"
    )


def test_assign_group_valueerror_when_unmatched():
    ap = helpers.make_action_parser()
    with pytest.raises(ValueError, match="not matched by any"):
        ap.assign_group(
            "totallyunknown_r",
            ANATOMICAL_TORSO_REGEX_MAP,
            {},
            singleton_fallback=False,
        )


########################################
#          Canonical granules          #
########################################


def test_get_canonical_granules_dispatch():
    assert get_canonical_granules("anatomical") is CANONICAL_GRANULES_ANATOMICAL
    assert get_canonical_granules("functional") is CANONICAL_GRANULES_FUNCTIONAL
    assert get_canonical_granules("hybrid") is CANONICAL_GRANULES_HYBRID
    # unknown -> falls back to hybrid
    assert get_canonical_granules("nonsense") is CANONICAL_GRANULES_HYBRID


def test_canonical_granule_contents():
    a = get_canonical_granules("anatomical")
    assert a["pectoralis"] == ["PECM1", "PECM2", "PECM3"]
    assert a["deltoids"] == ["DELT1", "DELT2", "DELT3"]
    h = get_canonical_granules("hybrid")
    assert h["hip_flexors"] == ["iliacus", "psoas", "recfem", "sart"]
    f = get_canonical_granules("functional")
    assert f["hip_extensors"] == [
        "bflh",
        "glmax1",
        "glmax2",
        "glmax3",
        "semimem",
        "semiten",
    ]


########################################
#           Muscle grouping            #
########################################


@pytest.mark.parametrize("strategy", ["anatomical", "functional", "hybrid"])
def test_get_muscle_grouping_matches_golden(strategy, goldens):
    ap = helpers.make_action_parser()
    mg = ap.get_muscle_grouping(strategy)
    g = goldens["action_parser"][strategy]

    assert mg.n_groups == g["n_groups"]
    assert mg.n_muscles == g["n_muscles"]
    assert mg.group_order == g["group_order"]
    assert mg.muscle_to_group == g["muscle_to_group"]
    assert mg.granule_base_map == g["granule_base_map"]

    # invariants
    assert mg.group_order == sorted(mg.group_order)
    assert set(mg.muscle_to_group.keys()) == set(helpers.MUSCLE_NAMES)
    total = sum(len(v) for v in mg.groups.values())
    assert total == len(helpers.MUSCLE_NAMES)


def test_get_muscle_grouping_unknown_strategy_raises():
    ap = helpers.make_action_parser()
    with pytest.raises(ValueError, match="Unknown strategy"):
        ap.get_muscle_grouping("bogus")


def test_anatomical_singleton_fallback_muscles():
    """Muscles not in any map are grouped as singletons under anatomical."""
    ap = helpers.make_action_parser(helpers.ANATOMICAL_ONLY_MUSCLE_NAMES)
    mg = ap.get_muscle_grouping("anatomical")
    assert mg.muscle_to_group["somecustom_r"] == "somecustom_r"
    assert mg.muscle_to_group["another"] == "another"
    assert mg.muscle_to_group["PECM1_r"] == "pectoralis_r"
