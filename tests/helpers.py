"""
Shared deterministic fixtures for Arnold characterization tests.

These builders are imported by both the golden-generation script
(tests/golden/generate_goldens.py) and the test modules so that the exact
same synthetic inputs are used in both places.

Nothing here touches the network, wandb, CUDA, git submodules, or any
expert wrapper. Everything is CPU-only and deterministic.
"""


########################################
#          Synthetic anatomy           #
########################################

# Body names (root first, mixed sided/center). Kept small but structurally
# realistic for the ObservationParser.
BODY_NAMES: list[str] = [
    "root",
    "pelvis",
    "femur_r",
    "femur_l",
    "tibia_r",
    "tibia_l",
    "calcn_r",
    "calcn_l",
    "toes_r",
    "toes_l",
]

# Muscle names chosen so that EVERY muscle matches a torso-regex or a limb-map
# entry under all three strategies (anatomical / functional / hybrid), i.e. no
# strategy needs singleton_fallback to succeed.  Sided torso fascicles + sided
# limb muscles.
MUSCLE_NAMES: list[str] = [
    # torso fascicles (match ANATOMICAL/FUNCTIONAL torso regex)
    "PECM1_r", "PECM2_r", "PECM3_r",
    "DELT1_l", "DELT2_l", "DELT3_l",
    "LAT1_r", "LAT2_r", "LAT3_r",
    "glmax1_l", "glmax2_l", "glmax3_l",
    # limb muscles (match limb dict maps)
    "psoas_r", "iliacus_r", "recfem_r",
    "vaslat_l", "vasmed_l", "vasint_l",
    "soleus_r", "gasmed_r", "tibant_r",
    "TRIlong_l", "TRIlat_l", "BIClong_l",
]

# A muscle set that only works for the "anatomical" strategy (needs the
# singleton fallback because these bases are not in any limb map).
ANATOMICAL_ONLY_MUSCLE_NAMES: list[str] = [
    "PECM1_r", "PECM2_r",
    "somecustom_r", "somecustom_l",   # unknown -> singleton fallback
    "another",                        # center, unknown -> singleton fallback
]

PROPRIOCEPTIVE_INPUTS: list[str] = [
    "root_height",
    "root_tilt",
    "local_body_pos",
    "local_body_rot",
    "local_body_vel",
    "local_body_ang_vel",
    "muscle_len",
    "muscle_vel",
    "muscle_force",
    "feet_contacts",
]

TASK_INPUTS: list[str] = [
    "diff_local_body_pos",
    "diff_local_vel",
    "local_ref_body_pos",
    "diff_muscle_len",
    "diff_muscle_vel",
]

TRACK_BODIES: list[str] = ["pelvis", "femur_r", "femur_l"]

HISTORY_LEN = 5
EMBED_DIM = 16


def make_observation_parser(history_len: int = HISTORY_LEN):
    """Builds a deterministic ObservationParser from the synthetic anatomy."""
    from arnold.observation_parser import ObservationParser

    return ObservationParser(
        body_names=BODY_NAMES,
        muscle_names=MUSCLE_NAMES,
        proprioceptive_inputs=PROPRIOCEPTIVE_INPUTS,
        task_inputs=TASK_INPUTS,
        track_bodies=TRACK_BODIES,
        num_bodies=len(BODY_NAMES),
        num_muscles=len(MUSCLE_NAMES),
        history_len=history_len,
    )


def make_action_parser(muscle_names: list[str] = None):
    from arnold.action_parser import ActionParser

    if muscle_names is None:
        muscle_names = MUSCLE_NAMES
    return ActionParser(muscle_names)
