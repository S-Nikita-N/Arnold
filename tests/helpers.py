"""
Shared deterministic fixtures for MyoTrainer characterization tests.

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
    "PECM1_r",
    "PECM2_r",
    "PECM3_r",
    "DELT1_l",
    "DELT2_l",
    "DELT3_l",
    "LAT1_r",
    "LAT2_r",
    "LAT3_r",
    "glmax1_l",
    "glmax2_l",
    "glmax3_l",
    # limb muscles (match limb dict maps)
    "psoas_r",
    "iliacus_r",
    "recfem_r",
    "vaslat_l",
    "vasmed_l",
    "vasint_l",
    "soleus_r",
    "gasmed_r",
    "tibant_r",
    "TRIlong_l",
    "TRIlat_l",
    "BIClong_l",
]

# A muscle set that only works for the "anatomical" strategy (needs the
# singleton fallback because these bases are not in any limb map).
ANATOMICAL_ONLY_MUSCLE_NAMES: list[str] = [
    "PECM1_r",
    "PECM2_r",
    "somecustom_r",
    "somecustom_l",  # unknown -> singleton fallback
    "another",  # center, unknown -> singleton fallback
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
    from myotrainer.observation_parser import ObservationParser

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
    from myotrainer.action_parser import ActionParser

    if muscle_names is None:
        muscle_names = MUSCLE_NAMES
    return ActionParser(muscle_names)


########################################
#           Policy builders            #
########################################

# Shared by the golden generator and the policy tests so both construct the
# exact same seeded networks and drive them with the same synthetic obs.
# All CPU-only, tiny, deterministic. `expert_name` is always "legs".

POLICY_OBS_SEED = 100
POLICY_BATCH = 2


def make_policy_io():
    """Returns (obs_parser, action_parser, obs_timeseries) deterministically."""
    import torch

    op = make_observation_parser()
    ap = make_action_parser()
    torch.manual_seed(POLICY_OBS_SEED)
    obs = torch.randn(POLICY_BATCH, op.n_obs_elements, HISTORY_LEN)
    return op, ap, obs


def build_lattice(state_dim: int, action_dim: int):
    import torch

    from myotrainer.torch_model.policy_lattice import LatticePolicy

    torch.manual_seed(10)
    net = LatticePolicy(state_dim, action_dim, mlp_units=(16, 8))
    net.eval()
    return net


def build_moe(state_dim: int, action_dim: int):
    import torch

    from myotrainer.torch_model.policy_moe import MoELatticePolicy

    torch.manual_seed(11)
    net = MoELatticePolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        shared_units=(8,),
        shared_expert_units=(8,),
        num_experts=3,
        top_k=2,
        expert_units=(8,),
        gate_units=(8,),
        value_units=(8,),
    )
    net.eval()
    return net


def build_transformer(op):
    import torch

    from myotrainer.torch_model.transformer_policy import TransformerPolicy
    from myotrainer.torch_model.sensorimotor_vocabulary import (
        SensorimotorVocabulary,
    )

    torch.manual_seed(12)
    vocab = SensorimotorVocabulary(embed_dim=EMBED_DIM)
    net = TransformerPolicy(
        vocab,
        {"legs": op.get_body_groups("per_body")},
        history_len=HISTORY_LEN,
        embed_dim=EMBED_DIM,
        ff_dim=32,
        num_heads=2,
        num_enc_layers=1,
        num_act_dec_layers=1,
        num_val_dec_layers=1,
        max_action_dim=32,
        action_cov_rank=4,
    )
    net.eval()
    return net


def build_transformer_granulated(op, ap):
    import torch

    from myotrainer.torch_model.transformer_policy import TransformerPolicy
    from myotrainer.torch_model.sensorimotor_vocabulary import (
        SensorimotorVocabulary,
    )

    grouping = ap.get_muscle_grouping("hybrid")
    torch.manual_seed(14)
    vocab = SensorimotorVocabulary(embed_dim=EMBED_DIM)
    net = TransformerPolicy(
        vocab,
        {"legs": op.get_body_groups("per_body")},
        history_len=HISTORY_LEN,
        embed_dim=EMBED_DIM,
        ff_dim=32,
        num_heads=2,
        num_enc_layers=1,
        num_act_dec_layers=1,
        num_val_dec_layers=1,
        max_action_dim=32,
        action_cov_rank=4,
        action_granulation="hybrid",
        action_groupings={"legs": grouping},
        action_signatures_by_expert={"legs": ap.action_signatures},
    )
    net.eval()
    return net


def write_expert_checkpoints(ckpt_dir, state_dim: int, action_dim: int):
    """Writes two tiny seeded LatticePolicy checkpoints; returns their paths."""
    import os

    import torch

    from myotrainer.torch_model.policy_lattice import LatticePolicy

    paths = []
    for i in range(2):
        torch.manual_seed(20 + i)
        expert = LatticePolicy(state_dim, action_dim, mlp_units=(8,))
        path = os.path.join(str(ckpt_dir), f"expert_{i}.pth")
        torch.save({"policy": expert.state_dict()}, path)
        paths.append(path)
    return paths


def build_gate_moe(state_dim: int, action_dim: int, ckpt_dir):
    import torch

    from myotrainer.torch_model.policy_gate_moe import GateMoEPolicy

    paths = write_expert_checkpoints(ckpt_dir, state_dim, action_dim)
    torch.manual_seed(13)
    net = GateMoEPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        expert_checkpoints=paths,
        gate_units=(8,),
    )
    net.eval()
    return net
