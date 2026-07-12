"""
Валидация обученной Arnold policy.

Headless режим (по умолчанию для CUDA/MPS) — vectorized eval через
ArnoldTrainer: workers стэппают env на CPU, главный процесс делает
batched inference на GPU. Логика полностью переиспользуется
из trainer'а (без дублирования).

Render режим (headless=false) — последовательный single-env loop
с возможностью визуализации MuJoCo и задержкой между шагами.

Использование:
    # Headless (vectorized, быстро):
    uv run python -m arnold.validate_policy \
        device=cuda \
        run=validate \
        run.checkpoint=/path/to/model.pth \
        '+run/experts@run.experts.myo=myohuman' \
        learning.policy=lattice

    # Render (single env):
    uv run mjpython -m arnold.validate_policy \
        run=validate \
        run.checkpoint=/path/to/model.pth \
        run.headless=false \
        '+run/experts@run.experts.myo=myohuman' \
        device=mps

Параметры (run=validate):
    run.checkpoint:      путь к .pth (обязательный)
    run.headless:        true (vectorized) / false (render single env)
    run.num_motions:     количество движений (-1 = все, только render-режим)
    run.sleep_per_step:  задержка между шагами, сек (только render-режим)
    run.fast_eval:       true — не считать std/value в get_action (render-режим)
    run.mode:            "valid" (default) | "train" — какой
                         набор motions использовать
"""

import time
import logging

import hydra
import torch
import numpy as np

from tqdm import tqdm
from omegaconf import DictConfig

from arnold.profiler import Profiler
from arnold.action_parser import ActionParser
from arnold.observation_parser import ObservationParser


logger = logging.getLogger(__name__)


########################################
#           Vectorized eval            #
########################################


def validate_vectorized(cfg: DictConfig) -> dict:
    """
    Vectorized validation: создаёт ArnoldTrainer (он загружает чекпоинт
    и поднимает valid_experts) и вызывает trainer._evaluate_expert_vectorized.

    Дефолты trainer-полей лежат в cfg/run/validate.yaml.
    """
    from omegaconf import OmegaConf
    from arnold.trainer import ArnoldTrainer

    OmegaConf.set_struct(cfg, False)
    cfg.resume_checkpoint = str(cfg.run.checkpoint)
    cfg.use_wandb = False
    cfg.no_log = True
    if "exp_name" not in cfg:
        cfg.exp_name = "validate_run"

    trainer = ArnoldTrainer(cfg, device=cfg.device)

    mode = cfg.run.get("mode", "valid")
    if mode == "train":
        # Swap valid wrappers to train wrappers and put envs into eval mode
        # (same trick as mine_negatives / trainer.evaluate_train).
        for ctx in trainer.experts.values():
            ctx.wrapper.env.start_eval(im_eval=True)
        trainer.valid_experts = {
            n: ctx.wrapper for n, ctx in trainer.experts.items()
        }

    all_metrics: dict = {}
    try:
        for expert_name, ctx in trainer.experts.items():
            wrapper = trainer.valid_experts.get(expert_name)
            if wrapper is None:
                raise ValueError(
                    f"No valid wrapper for expert '{expert_name}'. "
                    f"Is run.eval_frequency > 0?",
                )
            metrics = trainer._evaluate_expert_vectorized(
                expert_name,
                ctx,
                wrapper,
                return_per_motion=False,
            )
            all_metrics[expert_name] = metrics
    finally:
        if mode == "train":
            for ctx in trainer.experts.values():
                ctx.wrapper.env.end_eval()

    _print_summary(all_metrics)
    return all_metrics


def _print_summary(all_metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    for expert_name, m in all_metrics.items():
        print(f"  [{expert_name}]")
        for k, v in m.items():
            print(
                f"    {k:30s} = {v:.4f}"
                if isinstance(v, float)
                else f"    {k:30s} = {v}",
            )
    print("=" * 60)


########################################
#           Sequential eval            #
########################################


def _create_policy(cfg, expert, parser):
    """Build policy by type (mirrors ArnoldTrainer.setup_policy dispatch)."""
    from omegaconf import OmegaConf

    policy_type = cfg.learning.policy
    history_len = cfg.learning.history_len
    tokenizer_granularity = cfg.learning.tokenizer_granularity

    if policy_type == "transformer":
        from arnold.torch_model.sensorimotor_vocabulary import (
            SensorimotorVocabulary,
        )
        from arnold.torch_model.transformer_policy import TransformerPolicy

        transformer_cfg = OmegaConf.to_container(
            cfg.learning.transformer,
            resolve=True,
        )
        action_granulation = transformer_cfg.pop("action_granulation", "none")
        action_decoder_layers = transformer_cfg.pop(
            "action_decoder_layers",
            None,
        )

        expert_config = list(cfg.run.experts.values())[0]
        expert_name = expert_config.type
        groups = parser.get_body_groups(tokenizer_granularity)
        vocab = SensorimotorVocabulary(embed_dim=transformer_cfg["embed_dim"])

        action_groupings = None
        action_sigs_by_expert = None
        if action_granulation != "none":
            ap = ActionParser.from_env(expert.env)
            action_groupings = {
                expert_name: ap.get_muscle_grouping(
                    strategy=action_granulation,
                ),
            }
            action_sigs_by_expert = {expert_name: ap.action_signatures}

        return TransformerPolicy(
            vocab=vocab,
            groups={expert_name: groups},
            history_len=history_len,
            max_action_dim=expert.action_dim,
            action_granulation=action_granulation,
            action_groupings=action_groupings,
            action_signatures_by_expert=action_sigs_by_expert,
            action_decoder_layers=action_decoder_layers,
            **transformer_cfg,
        )

    if policy_type == "lattice":
        from arnold.torch_model.policy_lattice import LatticePolicy

        lattice_cfg = OmegaConf.to_container(cfg.learning.lattice, resolve=True)
        return LatticePolicy(
            state_dim=parser.n_obs_elements,
            action_dim=expert.action_dim,
            **lattice_cfg,
        )

    if policy_type == "moe":
        from arnold.torch_model.policy_moe import MoELatticePolicy

        moe_cfg = OmegaConf.to_container(cfg.learning.moe, resolve=True)
        return MoELatticePolicy(
            state_dim=parser.n_obs_elements,
            action_dim=expert.action_dim,
            **moe_cfg,
        )

    if policy_type == "gate_moe":
        from arnold.torch_model.policy_gate_moe import GateMoEPolicy

        gate_cfg = OmegaConf.to_container(cfg.learning.gate_moe, resolve=True)
        return GateMoEPolicy(
            state_dim=parser.n_obs_elements,
            action_dim=expert.action_dim,
            expert_checkpoints=list(gate_cfg.get("expert_checkpoints", [])),
            gate_units=tuple(gate_cfg.get("gate_units", (1024, 512, 256))),
            gate_activation=gate_cfg.get("gate_activation", "silu"),
        )

    raise ValueError(
        f"Unknown policy: {policy_type}. "
        f"Supported: transformer, lattice, moe, gate_moe",
    )


def _load_policy(cfg, checkpoint_path, device, expert, parser):
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    policy = _create_policy(cfg, expert, parser)

    policy_state = checkpoint["policy"]
    if policy_state and next(iter(policy_state.keys()), "").startswith(
        "module.",
    ):
        policy_state = {
            k.replace("module.", "", 1): v for k, v in policy_state.items()
        }

    try:
        policy.load_state_dict(policy_state)
    except RuntimeError as e:
        logger.warning(
            f"Strict load failed: {e}; retrying without size-mismatched keys",
        )
        model_state = policy.state_dict()
        filtered = {}
        for k, v in policy_state.items():
            if (
                k in model_state
                and hasattr(model_state[k], "shape")
                and hasattr(v, "shape")
                and model_state[k].shape != v.shape
            ):
                continue
            filtered[k] = v
        policy.load_state_dict(filtered, strict=False)

    policy.to(device)
    policy.eval()
    n_params = sum(p.numel() for p in policy.parameters())
    epoch = checkpoint.get("epoch", "?")
    logger.info(
        f"Policy loaded ({cfg.learning.policy}). "
        f"Epoch: {epoch}, parameters: {n_params:,}",
    )
    return policy


def validate_sequential(cfg: DictConfig) -> dict:
    """Single-env loop with optional MuJoCo render; used when headless=false."""
    from arnold.trainer import create_expert_wrapper

    run = cfg.run
    device = torch.device(cfg.device)
    history_len = cfg.learning.history_len
    sleep_per_step = run.sleep_per_step
    fast_eval = run.fast_eval

    experts_list = list(cfg.run.experts.values())
    if len(experts_list) != 1:
        raise ValueError("validate_sequential supports single expert only")
    expert_config = experts_list[0]
    expert_name = expert_config.type

    env_mode = run.get("mode", "valid")
    logger.info(
        f"Loading expert environment (mode={env_mode}, headless=False)...",
    )
    expert = create_expert_wrapper(
        expert_config,
        mode=env_mode,
        overrides=["run.headless=false"],
    )
    logger.info(
        f"Expert loaded. Obs dim: {expert.obs_dim}, "
        f"Action dim: {expert.action_dim}",
    )

    parser = ObservationParser.from_env(expert.env, history_len=history_len)
    action_parser = ActionParser.from_env(expert.env)
    policy = _load_policy(cfg, str(run.checkpoint), device, expert, parser)

    profiler = Profiler(device=str(device) if device.type == "cuda" else None)
    profiler.time_enabled = True

    episode_rewards: list = []
    episode_lengths: list = []
    episode_imitation_losses: list = []
    episode_mpjpes: list = []
    episode_frame_coverages: list = []
    episode_successes: list = []

    num_motions = run.num_motions
    total_motions = expert.num_motions
    if num_motions > 0:
        total_motions = min(num_motions, total_motions)

    logger.info(f"Validating on {total_motions} motions (mode={env_mode})...")
    if env_mode != "train":
        expert.env.start_eval(im_eval=True)

    profile_done = False
    try:
        motion_iter = expert.forward_motions()
        pbar = tqdm(motion_iter, total=total_motions, desc="Evaluating")

        for motion_idx, _motion_id in enumerate(pbar):
            if num_motions > 0 and motion_idx >= num_motions:
                break

            obs, info = expert.reset()
            parser.reset(obs)
            ep_reward = 0.0
            ep_length = 0
            step_im_losses: list = []

            for _t in range(10000):
                obs_ts, obs_sigs = parser.get_observation(device)
                act_sigs = action_parser.action_signatures

                with profiler.section("get_action"), torch.no_grad():
                    action, _, _, _ = policy.get_action(
                        obs_ts,
                        obs_sigs,
                        act_sigs,
                        expert_name=expert_name,
                        deterministic=True,
                        return_std=not fast_eval,
                        return_value=not fast_eval,
                    )
                action_np = action.squeeze(0).cpu().numpy()

                if expert.has_expert:
                    expert_action = expert.get_expert_action(obs)
                    step_im_losses.append(
                        float(((action_np - expert_action) ** 2).mean()),
                    )

                with profiler.section("env_step"):
                    next_obs, reward, terminated, truncated, info = expert.step(
                        action_np,
                    )
                done = terminated or truncated

                ep_reward += reward
                ep_length += 1
                expert.env.render()
                if sleep_per_step > 0:
                    time.sleep(sleep_per_step)

                parser.update(next_obs)
                obs = next_obs

                if done:
                    if "mpjpe" in info:
                        episode_mpjpes.append(float(info["mpjpe"]))
                    if "frame_coverage" in info:
                        episode_frame_coverages.append(
                            float(info["frame_coverage"]),
                        )
                    if "success" in info:
                        episode_successes.append(float(info["success"]))
                    if not profile_done:
                        profile_done = True
                        logger.info(profiler.report())
                        profiler.time_enabled = False
                    break

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            if step_im_losses:
                episode_imitation_losses.append(np.mean(step_im_losses))

            postfix = {"reward": f"{ep_reward:.2f}", "length": ep_length}
            if step_im_losses:
                postfix["im_loss"] = f"{np.mean(step_im_losses):.4f}"
            pbar.set_postfix(postfix)
    finally:
        if env_mode != "train":
            expert.env.end_eval()

    metrics = {
        expert_name: {
            "eval/mean_reward": float(np.mean(episode_rewards))
            if episode_rewards
            else 0.0,
            "eval/std_reward": float(np.std(episode_rewards))
            if episode_rewards
            else 0.0,
            "eval/mean_length": float(np.mean(episode_lengths))
            if episode_lengths
            else 0.0,
        },
    }
    if episode_imitation_losses:
        metrics[expert_name]["eval/imitation_loss"] = float(
            np.mean(episode_imitation_losses),
        )
    if episode_mpjpes:
        metrics[expert_name]["eval/mpjpe"] = float(np.mean(episode_mpjpes))
    if episode_frame_coverages:
        metrics[expert_name]["eval/frame_coverage"] = float(
            np.mean(episode_frame_coverages),
        )
    if episode_successes:
        metrics[expert_name]["eval/success_rate"] = float(
            np.mean(episode_successes),
        )

    _print_summary(metrics)
    return metrics


########################################
#             Entry point              #
########################################


def validate(cfg: DictConfig) -> dict:
    """Dispatch: vectorized (headless) or sequential (render)."""
    run = cfg.run
    if not str(run.checkpoint or "").strip():
        raise ValueError(
            "run.checkpoint is required: "
            "python -m arnold.validate_policy run=validate "
            "run.checkpoint=path/to/model.pth",
        )

    headless = bool(run.headless)
    if headless:
        return validate_vectorized(cfg)
    return validate_sequential(cfg)


@hydra.main(config_path="../../cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    validate(cfg)


if __name__ == "__main__":
    main()
