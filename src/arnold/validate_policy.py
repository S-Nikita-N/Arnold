"""
Валидация обученной Arnold policy с возможностью визуализации.

Поддерживает все архитектуры: transformer, lattice, moe.

Использование:
    poetry run mjpython -m arnold.validate_policy \
        run=validate \
        run.checkpoint=/path/to/model.pth \
        run.headless=false \
        '+run/experts@run.experts.myo=myohuman' \
        device=mps

Параметры (run=validate, префикс run.*):
    run.checkpoint: путь к .pth (обязательный)
    run.headless: true/false — визуализация
    run.num_motions: количество движений (-1 = все)
    run.sleep_per_step: задержка между шагами (сек)
    run.fast_eval: true — не считать std/value в get_action (быстрее)
    device: cpu/cuda/mps (из корня конфига)
"""

import time
import logging

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from arnold.trainer import create_expert_wrapper
from arnold.observation_parser import ObservationParser
from arnold.action_parser import ActionParser
from arnold.profiler import Profiler


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
#  Policy creation — mirrors trainer's setup_policy dispatch
# ------------------------------------------------------------------

def create_policy(cfg: DictConfig, expert, parser: ObservationParser):
    """
    Создаёт policy по типу из cfg.learning.policy.
    Повторяет логику ArnoldTrainer.setup_policy() без зависимости от Trainer.
    """
    policy_type = cfg.learning.policy
    history_len = cfg.learning.history_len
    tokenizer_granularity = cfg.learning.tokenizer_granularity

    if policy_type == "transformer":
        from arnold.torch_model.sensorimotor_vocabulary import SensorimotorVocabulary
        from arnold.torch_model.transformer_policy import TransformerPolicy

        transformer_cfg = OmegaConf.to_container(cfg.learning.transformer, resolve=True)
        action_granulation = transformer_cfg.pop("action_granulation", "none")
        action_decoder_layers = transformer_cfg.pop("action_decoder_layers", None)

        expert_config = list(cfg.run.experts.values())[0]
        expert_name = expert_config.type
        groups = parser.get_body_groups(tokenizer_granularity)

        vocab = SensorimotorVocabulary(embed_dim=transformer_cfg["embed_dim"])

        action_groupings = None
        action_sigs_by_expert = None
        if action_granulation != "none":
            action_parser = ActionParser.from_env(expert.env)
            muscle_grouping = action_parser.get_muscle_grouping(strategy=action_granulation)
            action_groupings = {expert_name: muscle_grouping}
            action_sigs_by_expert = {expert_name: action_parser.action_signatures}

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

    elif policy_type == "lattice":
        from arnold.torch_model.policy_lattice import LatticePolicy

        state_dim = parser.n_obs_elements
        action_dim = expert.action_dim
        lattice_cfg = OmegaConf.to_container(cfg.learning.lattice, resolve=True)
        return LatticePolicy(state_dim=state_dim, action_dim=action_dim, **lattice_cfg)

    elif policy_type == "moe":
        from arnold.torch_model.policy_moe import MoELatticePolicy

        state_dim = parser.n_obs_elements
        action_dim = expert.action_dim
        moe_cfg = OmegaConf.to_container(cfg.learning.moe, resolve=True)
        return MoELatticePolicy(state_dim=state_dim, action_dim=action_dim, **moe_cfg)

    else:
        raise ValueError(
            f"Unknown policy: {policy_type}. Supported: transformer, lattice, moe"
        )


def load_policy(cfg: DictConfig, checkpoint_path: str, device: torch.device, expert, parser):
    """Создаёт policy нужной архитектуры и загружает веса из чекпоинта."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    policy = create_policy(cfg, expert, parser)

    policy_state = checkpoint["policy"]
    if policy_state and next(iter(policy_state.keys()), "").startswith("module."):
        policy_state = {k.replace("module.", "", 1): v for k, v in policy_state.items()}

    try:
        policy.load_state_dict(policy_state)
    except RuntimeError as e:
        logger.warning(f"Strict load failed: {e}")
        logger.info("Filtering checkpoint for size mismatches (cross-environment transfer)...")
        model_state = policy.state_dict()
        filtered = {}
        for k, v in policy_state.items():
            if (
                k in model_state
                and hasattr(model_state[k], 'shape')
                and hasattr(v, 'shape')
                and model_state[k].shape != v.shape
            ):
                continue
            filtered[k] = v
        policy.load_state_dict(filtered, strict=False)

    policy.to(device)
    policy.eval()

    n_params = sum(p.numel() for p in policy.parameters())
    epoch = checkpoint.get("epoch", "?")
    logger.info(f"Policy loaded ({cfg.learning.policy}). Epoch: {epoch}, parameters: {n_params:,}")

    return policy


# ------------------------------------------------------------------
#  Validation loop
# ------------------------------------------------------------------

def validate(cfg: DictConfig) -> dict:
    """
    Валидация политики на движениях из validation set.
    Поддерживает все архитектуры: transformer, lattice, moe.
    """
    run = cfg.run
    checkpoint = run.checkpoint
    if not checkpoint or not str(checkpoint).strip():
        raise ValueError(
            "checkpoint is required. Usage: python -m arnold.validate_policy "
            "run=validate run.checkpoint=path/to/model.pth"
        )

    device = torch.device(cfg.device)
    history_len = cfg.learning.history_len
    headless = run.headless
    sleep_per_step = run.sleep_per_step
    fast_eval = run.fast_eval

    # Expert env
    experts_list = list(cfg.run.experts.values())
    if len(experts_list) == 0:
        raise ValueError("cfg.run.experts is empty")
    if len(experts_list) > 1:
        raise ValueError("validate_policy supports single expert only")

    expert_config = experts_list[0]
    expert_name = expert_config.type

    env_mode = run.get("mode", "valid")
    logger.info(f"Loading expert environment (mode={env_mode}, headless={headless})...")
    overrides = [f"run.headless={str(headless).lower()}"]
    expert = create_expert_wrapper(expert_config, mode=env_mode, overrides=overrides)
    logger.info(f"Expert loaded. Obs dim: {expert.obs_dim}, Action dim: {expert.action_dim}")

    # Parser
    parser = ObservationParser.from_env(expert.env, history_len=history_len)
    action_parser = ActionParser.from_env(expert.env)
    logger.info(f"Parser: {parser.n_obs_elements} observation elements")

    # Policy
    policy = load_policy(cfg, str(checkpoint), device, expert, parser)

    # Profiler — time only, первый эпизод
    profiler = Profiler(device=str(device) if device.type == "cuda" else None)
    profiler.time_enabled = True

    # Metrics
    episode_rewards = []
    episode_lengths = []
    episode_imitation_losses = []
    episode_mpjpes = []
    episode_frame_coverages = []
    episode_successes = []

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

        for motion_idx, motion_id in enumerate(pbar):
            if num_motions > 0 and motion_idx >= num_motions:
                break

            obs, info = expert.reset()
            parser.reset(obs)

            episode_reward = 0.0
            episode_length = 0
            step_imitation_losses = []

            for t in range(10000):
                with profiler.section("get_observation"):
                    obs_ts, obs_sigs = parser.get_observation(device)
                    act_sigs = action_parser.action_signatures

                with profiler.section("get_action"):
                    with torch.no_grad():
                        action, _, value = policy.get_action(
                            obs_ts, obs_sigs, act_sigs,
                            expert_name=expert_name,
                            deterministic=True,
                            return_std=not fast_eval,
                            return_value=not fast_eval,
                        )

                action_np = action.squeeze(0).cpu().numpy()

                with profiler.section("get_expert_action"):
                    if expert.has_expert:
                        expert_action = expert.get_expert_action(obs)
                        im_loss = float(((action_np - expert_action) ** 2).mean())
                        step_imitation_losses.append(im_loss)

                with profiler.section("env_step"):
                    next_obs, reward, terminated, truncated, info = expert.step(action_np)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                with profiler.section("render"):
                    if not headless:
                        expert.env.render()
                        if sleep_per_step > 0:
                            time.sleep(sleep_per_step)

                with profiler.section("parser_update"):
                    parser.update(next_obs)
                obs = next_obs

                if done:
                    if "mpjpe" in info:
                        episode_mpjpes.append(float(info["mpjpe"]))
                    if "frame_coverage" in info:
                        episode_frame_coverages.append(float(info["frame_coverage"]))
                    if "success" in info:
                        episode_successes.append(float(info["success"]))

                    # Print profiler report after first episode
                    if not profile_done:
                        profile_done = True
                        logger.info(profiler.report())
                        profiler.time_enabled = False

                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if step_imitation_losses:
                episode_imitation_losses.append(np.mean(step_imitation_losses))

            postfix = {
                "reward": f"{episode_reward:.2f}",
                "length": episode_length,
            }
            if step_imitation_losses:
                postfix["im_loss"] = f"{np.mean(step_imitation_losses):.4f}"
            pbar.set_postfix(postfix)

    finally:
        if env_mode != "train":
            expert.env.end_eval()

    # Final metrics
    metrics = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
    }
    if episode_imitation_losses:
        metrics["imitation_loss"] = float(np.mean(episode_imitation_losses))
    if episode_mpjpes:
        metrics["mpjpe"] = float(np.mean(episode_mpjpes))
    if episode_frame_coverages:
        metrics["frame_coverage"] = float(np.mean(episode_frame_coverages))
    if episode_successes:
        metrics["success_rate"] = float(np.mean(episode_successes))

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Motions evaluated: {len(episode_rewards)}")
    print(f"  Mean reward:       {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
    print(f"  Mean length:       {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
    if "imitation_loss" in metrics:
        print(f"  Imitation loss:    {metrics['imitation_loss']:.6f}")
    if "mpjpe" in metrics:
        print(f"  MPJPE:             {metrics['mpjpe'] * 1000:.2f} mm")
    if "frame_coverage" in metrics:
        print(f"  Frame coverage:    {metrics['frame_coverage']:.3f}")
    if "success_rate" in metrics:
        print(f"  Success rate:      {metrics['success_rate']:.3f}")
    print("=" * 60)

    return metrics


@hydra.main(config_path="../../cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    validate(cfg)


if __name__ == "__main__":
    main()
