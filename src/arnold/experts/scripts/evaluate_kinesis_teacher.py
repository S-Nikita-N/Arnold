#!/usr/bin/env python3
"""
Evaluate Kinesis teacher on MyoHuman and collect BC dataset.

Outputs (in data/ directory):
    {split}.json   — per-motion + per-frame metrics
    {split}.npz    — BC dataset (obs, actions, masks)

Usage:
    poetry run python -m arnold.experts.scripts.evaluate_kinesis_teacher \
        --checkpoint src/arnold/experts/Kinesis/data/trained_models/legs_back/kinesis-moe-imitation/model.pth \
        --split test \
        --num-workers 8 \
        --device mps
"""

import os
import sys
import json
import time
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import OrderedDict

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import mujoco
from scipy.spatial.transform import Rotation as sRot
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure myohuman package is importable
# ---------------------------------------------------------------------------
_EXPERTS_DIR = Path(__file__).resolve().parent.parent
_MYOHUMAN_SRC = _EXPERTS_DIR / "Myohuman" / "src"
if str(_MYOHUMAN_SRC) not in sys.path:
    sys.path.insert(0, str(_MYOHUMAN_SRC))

from myohuman.env.myolegs_env import compute_self_observations  # noqa: E402
import myohuman.utils.np_transform_utils as npt_utils  # noqa: E402

logger = logging.getLogger(__name__)
fork_ctx = mp.get_context("fork")

METADATA_DIR = Path(__file__).resolve().parent.parent / "data"

ARM_BODIES = {"humerus_r", "radius_r", "lunate_r",
              "humerus_l", "radius_l", "lunate_l"}

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

# Arms in MyoHuman action vector (absent in Kinesis)
ARM_ACTION_START = 290
ARM_ACTION_END = 338

# Architecture (from Kinesis config)
GATE_UNITS = [1024, 512, 256]
EXPERT_UNITS = [2048, 1536, 1024, 1024, 512, 512]
ACTIVATION = "silu"


# ========================== Lightweight NN blocks ==========================

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
        self.norm = _RunningNorm(OBS_DIM)
        self.experts = nn.ModuleList()
        for _ in range(NUM_EXPERTS):
            self.experts.append(nn.Sequential(
                _RunningNorm(OBS_DIM),
                _MLP(OBS_DIM, EXPERT_UNITS, ACTIVATION),
                nn.Linear(EXPERT_UNITS[-1], ACTION_DIM),
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
    heading_exp = heading_inv[:, None, :].repeat(J, axis=1)

    diff_pos = (ref_body_pos - body_pos).reshape(-1, 3)
    obs["diff_local_body_pos"] = npt_utils.quat_rotate(
        heading_exp.reshape(-1, 4), diff_pos,
    )

    diff_vel = (ref_body_vel - body_vel).reshape(-1, 3)
    obs["diff_local_vel"] = npt_utils.quat_rotate(
        heading_exp.reshape(-1, 4), diff_vel,
    )

    local_ref = (ref_body_pos - root_pos[:, None, :]).reshape(-1, 3)
    obs["local_ref_body_pos"] = npt_utils.quat_rotate(
        heading_exp.reshape(-1, 4), local_ref,
    )

    return obs


# ========================== KinesisObservationAdapter ======================

class KinesisObservationAdapter:
    """Builds 453-dim Kinesis observation from a MyoHuman env instance."""

    def __init__(self, mh_body_names: list[str]):
        mh = mh_body_names
        self.kin_to_mh_body = np.array([
            mh.index(BODY_PROXY.get(name, name))
            for name in KINESIS_BODY_NAMES
        ], dtype=np.intp)

        self.tracked_mh_ids = np.array([
            mh.index(name) for name in KINESIS_TRACKED_BODIES
        ], dtype=np.intp)

    def build_obs(self, env) -> np.ndarray:
        """Extract state from ``env`` and return a 453-dim float32 vector."""
        pos_all = env.get_body_xpos()
        rot_all = env.get_body_xquat()
        vel_all = env.get_body_linear_vel()
        ang_all = env.get_body_angular_vel()

        idx = self.kin_to_mh_body
        body_pos = pos_all[idx][None]
        body_rot = rot_all[idx][None]
        body_vel = vel_all[idx][None]
        body_ang = ang_all[idx][None]

        obs_dict = compute_self_observations(body_pos, body_rot, body_vel, body_ang)

        q = env.mj_data.qpos
        euler = sRot.from_quat([q[4], q[5], q[6], q[3]]).as_euler("xyz")
        root_tilt = np.array([
            np.cos(euler[0]), np.sin(euler[0]),
            np.cos(euler[1]), np.sin(euler[1]),
        ], dtype=np.float32)

        proprio = [
            obs_dict["root_h_obs"].ravel(),
            root_tilt,
            obs_dict["local_body_pos"].ravel(),
            obs_dict["local_body_rot_obs"].ravel(),
            obs_dict["local_body_vel"].ravel(),
            obs_dict["local_body_ang_vel"].ravel(),
            env.get_touch(),
        ]

        tid = self.tracked_mh_ids
        bp_t = pos_all[None, :, :][..., tid, :]
        bv_t = vel_all[None, :, :][..., tid, :]
        root_rot_q = rot_all[None, 0]
        root_pos = pos_all[None, 0]

        cache = env.ref_motion_cache
        ref_pos = cache.xpos[..., tid, :]
        ref_vel = cache.body_vel[..., tid, :]

        task_dict = _compute_imitation_obs(
            root_pos, root_rot_q, bp_t, bv_t, ref_pos, ref_vel,
        )
        task = [
            task_dict["diff_local_body_pos"].ravel(),
            task_dict["diff_local_vel"].ravel(),
            task_dict["local_ref_body_pos"].ravel(),
        ]

        return np.concatenate(proprio + task, dtype=np.float32)


# ========================== KinesisTeacher =================================

class KinesisTeacher(nn.Module):
    """Frozen Kinesis legs_back MoE policy.

    Loads once on GPU.  ``forward_batch`` does batched MoE inference.
    ``kin_to_mh`` / ``action_mask`` handle the 290->338 action mapping.
    """

    def __init__(self, checkpoint_path: str, device: str | torch.device):
        super().__init__()

        self.norm = _RunningNorm(OBS_DIM)
        self.gate = nn.Sequential(
            _MLP(OBS_DIM, GATE_UNITS, ACTIVATION),
            nn.Linear(GATE_UNITS[-1], NUM_EXPERTS),
            nn.Softmax(dim=1),
        )
        self.experts = _Experts()

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.load_state_dict(ckpt["policy"], strict=True)
        logger.info(
            "KinesisTeacher: loaded epoch %d from %s",
            ckpt.get("epoch", "?"), checkpoint_path,
        )
        del ckpt

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        self.to(device)

        # kin[0:80] (legs) → mh[210:290], kin[80:290] (torso) → mh[0:210]
        self.register_buffer("kin_to_mh", torch.cat([
            torch.arange(210, 290, dtype=torch.long),
            torch.arange(0, 210, dtype=torch.long),
        ]))

        mask = torch.zeros(MYOHUMAN_ACTION_DIM)
        mask[:290] = 1.0
        self.register_buffer("action_mask", mask)

    @torch.no_grad()
    def forward_batch(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched MoE inference.

        Args:
            obs: (B, 453) Kinesis-format observations on same device.

        Returns:
            actions_mh: (B, 338) actions in MyoHuman muscle order. Arm indices are 0.
            gate_max_prob: (B,) max softmax probability per sample.
        """
        B = obs.shape[0]

        normed = self.norm(obs)
        weights = self.gate(normed)
        gate_max_prob, expert_ids = weights.max(dim=1)

        actions_kin = torch.empty(B, ACTION_DIM, device=obs.device)
        for eid in range(NUM_EXPERTS):
            sel = expert_ids == eid
            if sel.any():
                actions_kin[sel] = self.experts.experts[eid](obs[sel])

        actions_mh = torch.zeros(B, MYOHUMAN_ACTION_DIM, device=obs.device)
        actions_mh[:, self.kin_to_mh] = actions_kin
        return actions_mh, gate_max_prob

    def get_action_mask(self) -> torch.Tensor:
        """(338,) float mask: 1 for teacher-controlled muscles, 0 for arms."""
        return self.action_mask


# ========================== Eval worker ====================================

def eval_worker(
    worker_id: int,
    obs_buf: torch.Tensor,
    action_buf: torch.Tensor,
    obs_ready,
    action_ready,
    stop_flag,
    motion_queue,
    result_queue,
    wrapper,
) -> None:
    """Env worker: builds obs, steps env, reports per-frame metrics + trajectory data."""
    np.random.seed(worker_id * 1337)

    env = wrapper.env
    adapter = KinesisObservationAdapter(env.body_names)

    tracked_ids = adapter.tracked_mh_ids

    # Match validate_cross_skeleton.py termination
    env.reset_bodies = [b for b in env.reset_bodies if b not in ARM_BODIES]
    env.reset_bodies_id = [env.body_names.index(b) for b in env.reset_bodies]
    env.termination_distance = 0.5
    env.per_body_termination_distance = None

    # Per-episode trajectory buffer
    ep_env_obs = []   # raw env obs (for BC dataset)
    ep_actions = []
    ep_frame_metrics = []

    def _load_next():
        try:
            mid = motion_queue.get_nowait()
        except Exception:
            return False
        env._current_eval_motion_id = int(mid)
        env._active_motion_ids = np.array([mid])
        return True

    def _reset_fill():
        wrapper.reset()
        env.mj_data.qpos[27:59] = 0.0
        mujoco.mj_forward(env.mj_model, env.mj_data)
        # Recompute env obs after arm zeroing
        raw_obs = env.compute_observations().astype(np.float32)
        # Adapter obs for teacher inference
        kin_obs = adapter.build_obs(env)
        obs_buf.copy_(torch.from_numpy(kin_obs))
        ep_env_obs.clear()
        ep_actions.clear()
        ep_frame_metrics.clear()
        ep_env_obs.append(raw_obs.copy())

    def _compute_frame_metrics():
        """Per-tracked-body position errors for current frame."""
        ref_state = env.get_state_from_motionlib_cache(
            env.cur_t * env.dt + env._motion_start_time
        )
        ref_pos = ref_state["xpos"][0, tracked_ids]
        sim_pos = env.get_body_xpos()[tracked_ids]
        per_body_err = np.linalg.norm(ref_pos - sim_pos, axis=-1)
        return {
            "mean_mpjpe": float(per_body_err.mean()),
            "max_mpjpe": float(per_body_err.max()),
            "per_body_mpjpe": per_body_err.tolist(),
        }

    try:
        if not _load_next():
            return
        _reset_fill()
        obs_ready.set()

        while True:
            action_ready.wait()
            action_ready.clear()
            if stop_flag.value:
                return

            action_np = action_buf.numpy().copy()
            ep_actions.append(action_np.copy())

            next_obs, reward, terminated, truncated, info = wrapper.step(action_np)

            frame_met = _compute_frame_metrics()
            ep_frame_metrics.append(frame_met)

            if terminated or truncated:
                result_queue.put(("episode", {
                    "motion_id": env._current_eval_motion_id,
                    "success": info.get("success", False),
                    "mpjpe": float(info.get("mpjpe", 0.0)),
                    "max_mpjpe": float(info.get("max_mpjpe", 0.0)),
                    "frame_coverage": float(info.get("frame_coverage", 0.0)),
                    "frame_metrics": ep_frame_metrics.copy(),
                    "trajectory": {
                        "obs": np.array(ep_env_obs, dtype=np.float32),
                        "actions": np.array(ep_actions, dtype=np.float32),
                    },
                }))
                if not _load_next():
                    return
                _reset_fill()
                obs_ready.set()
                continue

            # Raw env obs for BC dataset
            ep_env_obs.append(next_obs.astype(np.float32).copy())
            # Adapter obs for teacher inference
            kin_obs = adapter.build_obs(env)
            obs_buf.copy_(torch.from_numpy(kin_obs))
            obs_ready.set()
    except Exception as e:
        import traceback
        print(f"EvalWorker {worker_id} failed: {e}")
        traceback.print_exc()
    finally:
        result_queue.put(("done", worker_id))


# ========================== Main eval loop =================================

def run_eval(args):
    from arnold.experts.myohuman_wrapper import MyoHumanWrapper

    device = torch.device(args.device)

    mode = "valid" if args.split == "test" else "train"
    overrides = [
        "run.headless=true",
        "run.im_eval=true",
        "run.test=true",
        "run.random_start=false",
    ]

    wrapper = MyoHumanWrapper(
        checkpoint_epoch=0,
        device="cpu",
        overrides=overrides,
        mode=mode,
    )
    env = wrapper.env

    teacher = KinesisTeacher(args.checkpoint, device=device)
    action_mask = teacher.get_action_mask().cpu().numpy()

    total_motions = len(env._all_motion_ids)
    n_workers = min(args.num_workers, total_motions)

    logger.info("=" * 60)
    logger.info("Kinesis Teacher Evaluation + Dataset Collection")
    logger.info("  checkpoint: %s", args.checkpoint)
    logger.info("  split:      %s", args.split)
    logger.info("  motions:    %d", total_motions)
    logger.info("  workers:    %d", n_workers)
    logger.info("  device:     %s", device)
    logger.info("=" * 60)

    motion_queue = fork_ctx.Queue()
    for mid in env._all_motion_ids:
        motion_queue.put(int(mid))

    result_queue = fork_ctx.Queue()
    stop_flag = fork_ctx.Value("i", 0)

    class _WH:
        __slots__ = ("obs_buf", "action_buf", "obs_ready", "action_ready", "proc", "active")

    workers = []
    for w_idx in range(n_workers):
        wh = _WH()
        wh.obs_buf = torch.zeros(OBS_DIM, dtype=torch.float32).share_memory_()
        wh.action_buf = torch.zeros(MYOHUMAN_ACTION_DIM, dtype=torch.float32).share_memory_()
        wh.obs_ready = fork_ctx.Event()
        wh.action_ready = fork_ctx.Event()
        wh.active = True
        wh.proc = fork_ctx.Process(
            target=eval_worker,
            args=(
                w_idx + 1,
                wh.obs_buf, wh.action_buf,
                wh.obs_ready, wh.action_ready,
                stop_flag, motion_queue, result_queue,
                wrapper,
            ),
            daemon=True,
        )
        workers.append(wh)

    for wh in workers:
        wh.proc.start()

    per_motion = []
    pbar = tqdm(total=total_motions, desc="KinesisTeacher", unit="ep")
    workers_done = 0

    with torch.no_grad():
        while workers_done < n_workers:
            while True:
                try:
                    msg = result_queue.get_nowait()
                    if msg[0] == "episode":
                        per_motion.append(msg[1])
                        pbar.update(1)
                    elif msg[0] == "done":
                        workers_done += 1
                except Exception:
                    break

            if workers_done >= n_workers:
                break

            active = []
            for wh in workers:
                if not wh.active:
                    continue
                if wh.obs_ready.wait(timeout=0.5):
                    wh.obs_ready.clear()
                    active.append(wh)
                elif not wh.proc.is_alive():
                    wh.active = False

            if not active:
                continue

            batch_obs = torch.stack([wh.obs_buf.clone() for wh in active]).to(device)
            actions_mh, _ = teacher.forward_batch(batch_obs)
            actions_cpu = actions_mh.cpu()

            for i, wh in enumerate(active):
                wh.action_buf.copy_(actions_cpu[i])
                wh.action_ready.set()

    pbar.close()

    # Cleanup workers
    stop_flag.value = 1
    for wh in workers:
        wh.action_ready.set()
    deadline = time.time() + 10
    while workers_done < n_workers and time.time() < deadline:
        try:
            msg = result_queue.get(timeout=0.5)
            if msg[0] == "done":
                workers_done += 1
            elif msg[0] == "episode":
                per_motion.append(msg[1])
        except Exception:
            pass
    for wh in workers:
        wh.proc.join(timeout=3)
        if wh.proc.is_alive():
            wh.proc.terminate()

    # ── Aggregate metrics ──────────────────────────────────────
    successes = [r["success"] for r in per_motion]
    mpjpes = [r["mpjpe"] for r in per_motion]
    coverages = [r["frame_coverage"] for r in per_motion]
    metrics = {
        "success_rate": float(np.mean(successes)) if successes else 0,
        "mpjpe": float(np.mean(mpjpes)) if mpjpes else 0,
        "frame_coverage": float(np.mean(coverages)) if coverages else 0,
    }

    logger.info(
        "Result (%d workers): success=%.1f%%, mpjpe=%.1fmm, coverage=%.3f",
        n_workers,
        metrics["success_rate"] * 100,
        metrics["mpjpe"] * 1000,
        metrics["frame_coverage"],
    )

    # ── Save JSON (metrics + per-frame stats) ─────────────────
    out_dir = Path(args.output_dir) if args.output_dir else METADATA_DIR
    os.makedirs(str(out_dir), exist_ok=True)

    json_results = []
    for r in per_motion:
        json_results.append({
            "motion_id": r["motion_id"],
            "success": r["success"],
            "mpjpe": r["mpjpe"],
            "max_mpjpe": r["max_mpjpe"],
            "frame_coverage": r["frame_coverage"],
            "frame_metrics": r["frame_metrics"],
        })

    json_path = out_dir / f"{args.split}.json"
    with open(json_path, "w") as f:
        json.dump({
            "meta": {
                "checkpoint": str(args.checkpoint),
                "split": args.split,
                "total_motions": len(per_motion),
                **metrics,
            },
            "results": json_results,
        }, f, indent=2)
    logger.info("Metrics → %s", json_path)

    # ── Save NPZ dataset (obs, actions, action_mask, motion_ids) ──
    all_obs = []
    all_actions = []
    all_motion_ids = []
    all_frame_idx = []
    for r in per_motion:
        traj = r["trajectory"]
        obs_arr = traj["obs"]       # (T+1, obs_dim) — T+1 because includes initial obs
        act_arr = traj["actions"]   # (T, action_dim)
        T = act_arr.shape[0]
        # obs[t] → action[t] pairs (drop last obs which has no action)
        all_obs.append(obs_arr[:T])
        all_actions.append(act_arr)
        all_motion_ids.append(np.full(T, r["motion_id"], dtype=np.int32))
        all_frame_idx.append(np.arange(T, dtype=np.int32))

    dataset = {
        "obs": np.concatenate(all_obs, axis=0),
        "actions": np.concatenate(all_actions, axis=0),
        "action_mask": action_mask.astype(np.float32),
        "motion_ids": np.concatenate(all_motion_ids, axis=0),
        "frame_idx": np.concatenate(all_frame_idx, axis=0),
    }
    npz_path = out_dir / f"{args.split}.npz"
    np.savez_compressed(str(npz_path), **dataset)
    logger.info(
        "Dataset → %s (%d frames, obs=%s, actions=%s)",
        npz_path, dataset["obs"].shape[0],
        dataset["obs"].shape, dataset["actions"].shape,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for .json/.npz (default: experts/data/)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    run_eval(args)


if __name__ == "__main__":
    main()
