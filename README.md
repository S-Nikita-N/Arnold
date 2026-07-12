# MyoTrainer — training muscle policies for whole-body motion imitation

[![CI](https://github.com/nikswir/MyoTrainer/actions/workflows/ci.yml/badge.svg)](https://github.com/nikswir/MyoTrainer/actions/workflows/ci.yml)

MyoTrainer trains neural-network **muscle-control policies** that imitate human
motion in the **[MyoHuman](https://github.com/nikswir/Myohuman)** environment —
a whole-body musculoskeletal model with **338 Hill-type muscles** and
**86 degrees of freedom**. It combines **PPO** reinforcement learning,
**On-Policy Behavior Cloning (OBC)** distillation, a scalable **muscle
transformer** architecture, and a curriculum that bootstraps whole-body control
from an existing locomotion policy.

> **MyoTrainer is the *trainer*.** The body model, imitation task and motion
> data are provided by **[MyoHuman](https://github.com/nikswir/Myohuman)**,
> which MyoTrainer consumes as a submodule. This repo is the algorithms and
> network architectures that learn to act in it.

> **Built on Arnold.** The transformer policy is a reimplementation and
> adaptation of
> Chiappa, An, Simos, Li, Mathis — *Arnold: A Generalist Muscle Transformer
> Policy*, 2025 ([arXiv:2508.18066](https://arxiv.org/abs/2508.18066)) — scaled
> from MyoSuite's small models to the 338-muscle full-body MyoHuman, with a
> hierarchical action decoder, observation/action granulation, a transformer
> adaptation of LATTICE exploration, and a multi-stage training pipeline.

<!-- TODO: demo GIF of a trained expert imitating a reference clip -->
<p align="center">
  <em>Demo GIF — coming soon.</em>
</p>

## The idea

Learning to control 338 muscles **from scratch** is impractical: standard PPO —
with either an MLP or a transformer policy — stalls in a *standing* local
optimum (it tracks the arms but never commits to a step), and the transformer
is also ~10× more expensive per epoch. MyoTrainer instead **transfers** an
existing locomotion skill and grows it into whole-body control, then
**distills** the result into a scalable generalist transformer.

### 1. Three-stage expert (curriculum)

A single MLP actor is carried through three stages that progressively relax
toward the full MyoHuman task, reusing the pretrained
**[Kinesis](https://arxiv.org/abs/2503.14637)** (Simos et al., 2025) locomotion policy (290 leg+torso
muscles):

| Stage | Method | Arms | Result (SR / FC / MPJPE) |
| ----- | ------ | ---- | ------------------------ |
| 1 | **OBC** distill Kinesis into MyoHuman | frozen at zero, termination off | 0.08 / 0.35 / 0.104 m |
| 2 | **PPO**, relaxed arm requirements | learned, soft termination & reward | 0.36 / 0.60 / 0.090 m |
| 3 | **PPO**, full target task | full weight, 0.15 m termination | **0.45 / 0.68 / 0.061 m** |

*SR = success rate, FC = frame coverage, MPJPE = mean per-joint position error.*

### 2. Distill into a muscle transformer

The Stage-3 MLP expert is distilled (again via OBC) into an **encoder–decoder
transformer** whose tokens carry a **sensorimotor vocabulary** — each
observation and muscle is described by role tokens (side, anatomical structure,
signal type). To scale attention from 800+ sensory and 338 motor tokens, MyoTrainer
**granulates** tokens:

- **observation granulation** — `none` / by-specification / **by-body**;
- **action granulation** — `none` / anatomical / functional / **hybrid**
  (muscles grouped into ~30 granules by joint function and anatomy);
- **hierarchical decoder** — group-to-group (G2G) attention blocks plus
  within-group (WG) blocks and/or MLP **expansion heads** that unfold each
  group token back into individual muscle activations;
- **LATTICE** — latent time-correlated exploration adapted to the tokenized
  decoder (direct or projected covariance factor).

The strongest distillation configs use **by-body observations + functional/hybrid
action granulation + a G2G+MLP decoder**; a `G2G×5 + WG×1` decoder is the
cheapest per epoch. In the OBC setting, LATTICE gave no convergence gain.

### 3. Ensemble (mixture of experts)

Beyond the first expert, **hard-negative mining** yields specialists for the
clips the current policy imitates worst, and a **gate network** (frozen experts
+ trainable categorical router) combines them — the path toward a single
generalist policy covering the whole motion set.

## Setup

Requires **Git**, **[Git LFS](https://git-lfs.com/)** and
**[uv](https://docs.astral.sh/uv/)** (installs Python 3.12+ itself).

```bash
git lfs install                                             # once per machine
git clone --recurse-submodules <repo-url> MyoTrainer        # + Kinesis, MyoHuman
cd MyoTrainer
git lfs pull                                                # heavy assets under downloads/
./scripts/install.sh                                        # uv sync into .venv
./scripts/setup_experts.sh                                  # patch submodules, stage assets, fetch Kinesis model
```

`setup_experts.sh` initializes submodules, pulls their LFS files (including
MyoHuman), patches and stages the Kinesis assets into
`src/myotrainer/experts/Kinesis/data/`, and downloads the `kinesis-moe-imitation`
expert from Hugging Face.

Already cloned without submodules?
`git submodule update --init --recursive && git lfs pull`.

## Usage

Training is configured with [Hydra](https://hydra.cc) and driven through one
entry point; experts are added as config groups.

```bash
# Stage 1 — OBC distillation from the Kinesis locomotion expert
uv run python -m myotrainer.train_myotrainer \
    '+run/experts@run.experts.kin=kinesis' \
    exp_name=obc_kinesis

# PPO directly in MyoHuman
uv run python -m myotrainer.train_myotrainer \
    '+run/experts@run.experts.myo=myohuman' \
    run.experts.myo.learning.ppo_weight=1.0 \
    run.experts.myo.learning.obc_imitation_weight=0 \
    run.num_threads=15 \
    exp_name=ppo_myohuman

# Multi-GPU (DataParallel)
uv run python -m myotrainer.train_myotrainer \
    device_ids=[0,1] '+run/experts@run.experts.kin=kinesis' exp_name=obc_2gpu
```

Validate a trained policy and mine hard negatives:

```bash
uv run python -m myotrainer.validate_policy   # roll out & score imitation metrics
uv run python -m myotrainer.mine_negatives    # select worst-imitated clips
```

## Repo layout

```text
src/myotrainer/
├── train_myotrainer.py    training entry point (single- & multi-expert)
├── trainer.py             MyoTrainer: PPO + OBC loop, logging, checkpoints
├── torch_model/
│   ├── transformer_policy.py     encoder–decoder muscle transformer
│   ├── policy_lattice.py         MLP policy with LATTICE exploration
│   ├── policy_moe.py / policy_gate_moe.py   mixture-of-experts + gate router
│   ├── sensorimotor_vocabulary.py, body_tokenizer.py, action_tokenizer.py
│   └── sensory_encoder.py, normalization.py, mlp.py
├── experts/               wrappers for Kinesis, MyoHuman, MyoKin + submodules
├── observation_parser.py / action_parser.py   granulation / token layout
├── mine_negatives.py, validate_policy.py       hard-negative mining, evaluation
└── memory.py, profiler.py, logger.py, wandb_logger.py
cfg/                       Hydra config (config.yaml, learning/, run/, run/experts/)
downloads/Kinesis_assets/  SMPL, motion dicts, initial poses (Git LFS)
scripts/                   install.sh, setup_experts.sh
tests/                     characterization tests (+ stage-2 device gate)
```

Submodules: **Kinesis** and **MyoHuman** (the environment). More on experts:
[src/myotrainer/experts/README.md](src/myotrainer/experts/README.md).

## How it relates to MyoHuman

| | [MyoHuman](https://github.com/nikswir/Myohuman) | MyoTrainer (this repo) |
| --- | --- | --- |
| Role | the **environment** — body, task, data | the **trainer** — policies & algorithms |
| Provides | `MyoHumanIm`, MJCF model, retargeted motions | muscle transformer, PPO / OBC pipeline, MoE |
| Consumed as | Git submodule | — |

## References

- **Arnold** — Chiappa et al., *A Generalist Muscle Transformer Policy*, 2025,
  [arXiv:2508.18066](https://arxiv.org/abs/2508.18066).
- **Kinesis** — Simos et al., *Reinforcement Learning-Based Motion Imitation
  for Physiologically Plausible Musculoskeletal Motor Control*, 2025,
  [arXiv:2503.14637](https://arxiv.org/abs/2503.14637) — locomotion imitation +
  hard-negative mining / MoE for a legs+torso musculoskeletal model.
- **LATTICE** — Chiappa et al., 2023 — latent time-correlated exploration.
- **PPO** — Schulman et al., 2017. **DeepMimic** — Peng et al., 2018.
- **MyoSuite** — Caggiano et al., 2022.

## Development

The engineering workflow — toolchain (uv), the pre-commit gate, two-stage
tests, CI — is documented in [AGENTS.md](AGENTS.md).

```bash
uv run pre-commit run --all-files   # lint, format, types, style checks
uv run pytest                       # stage-1 (fast, CPU) tests
RUN_STAGE2=1 uv run pytest          # + device-parity tests
```
