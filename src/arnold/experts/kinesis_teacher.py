"""
Kinesis Teacher: frozen MoE policy for knowledge distillation.

Provides:
  - KinesisObservationAdapter: builds 453-dim Kinesis obs from MyoHuman env state
  - KinesisTeacher: loads legs_back checkpoint, batched GPU inference, action mapping
"""

import sys
import logging
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as sRot

# ---------------------------------------------------------------------------
# Ensure myohuman package is importable (same pattern as myohuman_wrapper.py)
# ---------------------------------------------------------------------------
_EXPERTS_DIR = Path(__file__).resolve().parent
_MYOHUMAN_SRC = _EXPERTS_DIR / "Myohuman" / "src"
if str(_MYOHUMAN_SRC) not in sys.path:
    sys.path.insert(0, str(_MYOHUMAN_SRC))

from myohuman.env.myolegs_env import compute_self_observations  # noqa: E402
import myohuman.utils.np_transform_utils as npt_utils  # noqa: E402

logger = logging.getLogger(__name__)

# ========================== Constants ======================================

# 25 bodies in Kinesis legs_back model (excl. world), MuJoCo order
KINESIS_BODY_NAMES = [
    "root", "sacrum", "lumbar5", "Abdomen", "lumbar4", "lumbar3",
    "lumbar2", "lumbar1", "torso", "head_attach", "neck", "head",
    "pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r",
    "patella_r", "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l",
    "patella_l",
]

# 8 tracked bodies (MYOLEG_ABS_TRACKED_BODIES)
KINESIS_TRACKED_BODIES = [
    "root", "head", "tibia_l", "tibia_r",
    "talus_l", "talus_r", "toes_l", "toes_r",
]

# Bodies absent from MyoHuman → proxy body
BODY_PROXY = {"head_attach": "neck"}

OBS_DIM = 453
ACTION_DIM = 290
NUM_EXPERTS = 3
MYOHUMAN_ACTION_DIM = 338

# Arms in MyoHuman action vector (absent in Kinesis). Used by arm regularizer.
ARM_ACTION_START = 290
ARM_ACTION_END = 338

# Architecture (from Kinesis config)
GATE_UNITS = [1024, 512, 256]
EXPERT_UNITS = [2048, 1536, 1024, 1024, 512, 512]
ACTIVATION = "silu"


# ========================== Lightweight NN blocks ==========================
# Re-implemented to avoid importing Kinesis internals.

class _RunningNorm(nn.Module):
    """Observation normalizer (frozen — only uses loaded statistics)."""

    def __init__(self, dim, clip=5.0):
        super().__init__()
        self.clip = clip
        self.register_buffer("n", torch.tensor(0, dtype=torch.long))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.zeros(dim))
        self.register_buffer("std", torch.zeros(dim))

    def forward(self, x):
        if self.n > 0:
            x = x - self.mean
            x = x / (self.std + 1e-8)
            if self.clip:
                x = torch.clamp(x, -self.clip, self.clip)
        return x


class _MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation="silu"):
        super().__init__()
        act = {"silu": nn.SiLU(), "tanh": nn.Tanh(), "relu": nn.ReLU()}
        self.activation = act[activation]
        self.affine_layers = nn.ModuleList()
        last = input_dim
        for h in hidden_dims:
            self.affine_layers.append(nn.Linear(last, h))
            last = h

    def forward(self, x):
        for layer in self.affine_layers:
            x = self.activation(layer(x))
        return x


class _Experts(nn.Module):
    """Container matching Kinesis `Experts` state-dict layout."""

    def __init__(self):
        super().__init__()
        self.norm = _RunningNorm(OBS_DIM)          # experts.norm.*
        self.experts = nn.ModuleList()              # experts.experts.*
        for _ in range(NUM_EXPERTS):
            self.experts.append(nn.Sequential(
                _RunningNorm(OBS_DIM),                          # .0
                _MLP(OBS_DIM, EXPERT_UNITS, ACTIVATION),        # .1
                nn.Linear(EXPERT_UNITS[-1], ACTION_DIM),        # .2
            ))


# ========================== Observation helpers ============================

def _compute_imitation_obs(root_pos, root_rot, body_pos, body_vel,
                           ref_body_pos, ref_body_vel):
    """Kinesis-compatible imitation observations (no muscle features).

    All inputs have batch dim B=1.  Returns OrderedDict with 3 keys,
    each ravelled to (8*3,) = 24-dim.
    """
    obs = OrderedDict()
    B, J, _ = body_pos.shape

    heading_inv = npt_utils.calc_heading_quat_inv(root_rot)
    heading_exp = heading_inv[:, None, :].repeat(J, axis=1)  # (B, J, 4)

    # diff body pos
    diff_pos = (ref_body_pos - body_pos).reshape(-1, 3)
    obs["diff_local_body_pos"] = npt_utils.quat_rotate(
        heading_exp.reshape(-1, 4), diff_pos,
    )

    # diff vel
    diff_vel = (ref_body_vel - body_vel).reshape(-1, 3)
    obs["diff_local_vel"] = npt_utils.quat_rotate(
        heading_exp.reshape(-1, 4), diff_vel,
    )

    # local ref body pos
    local_ref = (ref_body_pos - root_pos[:, None, :]).reshape(-1, 3)
    obs["local_ref_body_pos"] = npt_utils.quat_rotate(
        heading_exp.reshape(-1, 4), local_ref,
    )

    return obs


# ========================== KinesisObservationAdapter ======================

class KinesisObservationAdapter:
    """Builds 453-dim Kinesis observation from a MyoHuman env instance.

    Constructed once per worker (after fork); stateless after __init__.
    """

    def __init__(self, mh_body_names: list[str]):
        mh = mh_body_names

        # 25 Kinesis bodies → MyoHuman body_names indices
        self.kin_to_mh_body = np.array([
            mh.index(BODY_PROXY.get(name, name))
            for name in KINESIS_BODY_NAMES
        ], dtype=np.intp)

        # 8 tracked bodies → MyoHuman body_names indices
        self.tracked_mh_ids = np.array([
            mh.index(name) for name in KINESIS_TRACKED_BODIES
        ], dtype=np.intp)

    def build_obs(self, env) -> np.ndarray:
        """Extract state from ``env`` and return a 453-dim float32 vector."""
        # ---------- raw MyoHuman states ----------
        pos_all = env.get_body_xpos()           # (97, 3)
        rot_all = env.get_body_xquat()          # (97, 4)
        vel_all = env.get_body_linear_vel()     # (97, 3)
        ang_all = env.get_body_angular_vel()    # (97, 3)

        # ---------- 25-body subset (Kinesis order) ----------
        idx = self.kin_to_mh_body
        body_pos = pos_all[idx][None]           # (1, 25, 3)
        body_rot = rot_all[idx][None]           # (1, 25, 4)
        body_vel = vel_all[idx][None]           # (1, 25, 3)
        body_ang = ang_all[idx][None]           # (1, 25, 3)

        obs_dict = compute_self_observations(body_pos, body_rot, body_vel, body_ang)

        # root tilt (cos/sin of roll & pitch)
        q = env.mj_data.qpos
        euler = sRot.from_quat([q[4], q[5], q[6], q[3]]).as_euler("xyz")
        root_tilt = np.array([
            np.cos(euler[0]), np.sin(euler[0]),
            np.cos(euler[1]), np.sin(euler[1]),
        ], dtype=np.float32)

        # ---- proprioceptive (381 dim) ----
        proprio = [
            obs_dict["root_h_obs"].ravel(),         # 1
            root_tilt,                               # 4
            obs_dict["local_body_pos"].ravel(),      # 72  (25*3 - 3)
            obs_dict["local_body_rot_obs"].ravel(),  # 150 (25*6)
            obs_dict["local_body_vel"].ravel(),      # 75  (25*3)
            obs_dict["local_body_ang_vel"].ravel(),  # 75  (25*3)
            env.get_touch(),                         # 4
        ]

        # ---- task (72 dim) ----
        tid = self.tracked_mh_ids
        bp_t = pos_all[None, :, :][..., tid, :]         # (1, 8, 3)
        bv_t = vel_all[None, :, :][..., tid, :]         # (1, 8, 3)
        root_rot_q = rot_all[None, 0]                   # (1, 4)
        root_pos = pos_all[None, 0]                     # (1, 3)

        cache = env.ref_motion_cache
        ref_pos = cache.xpos[..., tid, :]               # (1, 8, 3)
        ref_vel = cache.body_vel[..., tid, :]           # (1, 8, 3)

        task_dict = _compute_imitation_obs(
            root_pos, root_rot_q, bp_t, bv_t, ref_pos, ref_vel,
        )
        task = [
            task_dict["diff_local_body_pos"].ravel(),    # 24
            task_dict["diff_local_vel"].ravel(),         # 24
            task_dict["local_ref_body_pos"].ravel(),     # 24
        ]

        return np.concatenate(proprio + task, dtype=np.float32)


# ========================== KinesisTeacher =================================

class KinesisTeacher(nn.Module):
    """Frozen Kinesis legs_back MoE policy for teacher-guided training.

    Loads once on GPU.  ``forward_batch`` does batched MoE inference.
    ``kin_to_mh`` / ``action_mask`` handle the 290→338 action mapping.
    """

    def __init__(self, checkpoint_path: str, device: str | torch.device):
        super().__init__()

        # ---- reconstruct architecture (matches checkpoint keys) ----
        self.norm = _RunningNorm(OBS_DIM)
        self.gate = nn.Sequential(
            _MLP(OBS_DIM, GATE_UNITS, ACTIVATION),
            nn.Linear(GATE_UNITS[-1], NUM_EXPERTS),
            nn.Softmax(dim=1),
        )
        self.experts = _Experts()

        # ---- load weights ----
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.load_state_dict(ckpt["policy"], strict=True)
        logger.info(
            "KinesisTeacher: loaded epoch %d from %s",
            ckpt.get("epoch", "?"), checkpoint_path,
        )
        del ckpt

        # ---- freeze ----
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.to(device)

        # ---- action mapping ----
        # kin[0:80]  (legs)  → mh[210:290]
        # kin[80:290] (torso) → mh[0:210]
        self.register_buffer("kin_to_mh", torch.cat([
            torch.arange(210, 290, dtype=torch.long),
            torch.arange(0, 210, dtype=torch.long),
        ]))  # (290,)

        mask = torch.zeros(MYOHUMAN_ACTION_DIM)
        mask[:210] = 1.0    # torso (from kin[80:290])
        mask[210:290] = 1.0  # legs  (from kin[0:80])
        # mask[290:338] = 0  — arms, not in Kinesis
        self.register_buffer("action_mask", mask)  # (338,)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_batch(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched MoE inference.

        Args:
            obs: (B, 453) Kinesis-format observations on same device.

        Returns:
            actions_mh: (B, 338) actions in MyoHuman muscle order. Arm indices are 0.
            gate_max_prob: (B,) max softmax probability per sample — proxy for
                teacher confidence. Low values (~1/3) mean gate is indecisive.
        """
        B = obs.shape[0]

        # Gate → deterministic expert selection
        normed = self.norm(obs)
        weights = self.gate(normed)                    # (B, num_experts)
        gate_max_prob, expert_ids = weights.max(dim=1)  # (B,), (B,)

        # Run experts grouped for efficiency
        actions_kin = torch.empty(B, ACTION_DIM, device=obs.device)
        for eid in range(NUM_EXPERTS):
            sel = expert_ids == eid
            if sel.any():
                actions_kin[sel] = self.experts.experts[eid](obs[sel])

        # Map 290-dim Kinesis → 338-dim MyoHuman
        actions_mh = torch.zeros(B, MYOHUMAN_ACTION_DIM, device=obs.device)
        actions_mh[:, self.kin_to_mh] = actions_kin
        return actions_mh, gate_max_prob

    def get_action_mask(self) -> torch.Tensor:
        """(338,) float mask: 1 for teacher-controlled muscles, 0 for arms."""
        return self.action_mask
