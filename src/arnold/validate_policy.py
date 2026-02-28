"""
Валидация обученной Arnold policy с возможностью визуализации.

Использование:
    poetry run mjpython -m arnold.validate_policy \
        run=validate \
        run.checkpoint=/Users/nikita/Projects/diploma/arnold/data/trained_models/obc_run_A100_80GB_3/model_best_ep_length.pth \
        run.headless=false \
        device=mps

Параметры (run=validate, префикс run.*):
    run.checkpoint: путь к .pth (обязательный)
    run.headless: true/false — визуализация
    run.num_motions: количество движений (-1 = все)
    run.sleep_per_step: задержка между шагами (сек)
    device: cpu/cuda/mps (из корня конфига)
"""

import time
import logging

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from arnold.trainer import create_expert_wrapper
from arnold.observation_parser import ObservationParser
from arnold.torch_model.sensorimotor_vocabulary import SensorimotorVocabulary
from arnold.torch_model.transformer_policy import TransformerPolicy


logger = logging.getLogger(__name__)


def load_policy(
    checkpoint_path: str,
    device: torch.device,
    learning_cfg,
    parser: ObservationParser,
) -> tuple:
    """
    Загружает политику из чекпоинта.

    Returns:
        (policy, hyperparams) — политика и гиперпараметры из чекпоинта
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    embed_dim = learning_cfg.embed_dim
    ff_dim = learning_cfg.ff_dim
    num_heads = learning_cfg.num_heads
    num_enc_layers = learning_cfg.num_enc_layers
    num_act_dec_layers = learning_cfg.num_act_dec_layers
    num_val_dec_layers = learning_cfg.num_val_dec_layers
    dropout = learning_cfg.dropout
    history_len = learning_cfg.history_len

    granularity = learning_cfg.tokenizer_granularity
    groups = parser.get_body_groups(granularity)
    vocab = SensorimotorVocabulary(embed_dim=embed_dim)
    policy = TransformerPolicy(
        vocab=vocab,
        groups=groups,
        history_len=history_len,
        embed_dim=embed_dim,
        ff_dim=ff_dim,
        num_heads=num_heads,
        num_enc_layers=num_enc_layers,
        num_act_dec_layers=num_act_dec_layers,
        num_val_dec_layers=num_val_dec_layers,
        dropout=dropout,
    )

    policy.load_state_dict(checkpoint["policy"])
    policy.to(device)
    policy.eval()

    logger.info(f"Policy loaded. Parameters: {sum(p.numel() for p in policy.parameters()):,}")
    logger.info(
        f"  embed_dim={embed_dim}, ff_dim={ff_dim}, num_heads={num_heads}, "
        f"enc_layers={num_enc_layers}, act_dec_layers={num_act_dec_layers}, "
        f"val_dec_layers={num_val_dec_layers}, history_len={history_len}, "
        f"body_groups={len(groups)}"
    )

    return policy, {"history_len": history_len}


def validate(cfg: DictConfig) -> dict:
    """
    Валидация политики на движениях из validation set.
    Все параметры берутся из cfg (run=validate, learning, device).
    
    Returns:
        dict с метриками
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

    # Параметры run для валидации
    headless = run.headless
    sleep_per_step = run.sleep_per_step

    logger.info(f"Loading expert environment (valid mode, headless={headless})...")

    experts_list = list(cfg.run.experts.values())
    if len(experts_list) == 0:
        raise ValueError("cfg.run.experts is empty")
    expert_entry = experts_list[0]

    overrides = [f"run.headless={str(headless).lower()}"]
    expert = create_expert_wrapper(expert_entry, mode="valid", overrides=overrides)
    logger.info(f"Expert loaded. Obs dim: {expert.obs_dim}, Action dim: {expert.action_dim}")

    # Создаём парсер (нужен для body groups при создании policy)
    parser = ObservationParser.from_env(expert.env, history_len=history_len)
    logger.info(f"Parser: {parser.n_obs_elements} observation elements")

    # Загружаем политику (архитектура из cfg.learning, groups из parser)
    policy, hp = load_policy(checkpoint, device, cfg.learning, parser)
    
    # Собираем метрики
    episode_rewards = []
    episode_lengths = []
    episode_imitation_losses = []
    
    # Итерация по движениям
    num_motions = run.num_motions
    total_motions = expert.num_motions
    if num_motions > 0:
        total_motions = min(num_motions, total_motions)
    
    # Профилирование шага (накапливаем за первый эпизод)
    profile_times = {
        "get_observation": 0.0,
        "policy_get_action": 0.0,
        "get_expert_action": 0.0,
        "expert_step": 0.0,
        "render": 0.0,
        "parser_update": 0.0,
    }
    profile_steps = 0
    profile_done = False
    
    logger.info(f"Validating on {total_motions} motions...")
    expert.env.start_eval(im_eval=True)
    
    try:
        motion_iter = expert.forward_motions()
        pbar = tqdm(motion_iter, total=total_motions, desc="Evaluating")
        
        for motion_idx, motion_id in enumerate(pbar):
            if num_motions > 0 and motion_idx >= num_motions:
                break
            
            # Reset для текущего движения
            obs, info = expert.reset()
            parser.reset(obs)
            
            episode_reward = 0.0
            episode_length = 0
            step_imitation_losses = []
            
            for t in range(10000):
                t0 = time.perf_counter()
                obs_ts, obs_sigs = parser.get_observation(device)
                act_sigs = parser.action_signatures
                profile_times["get_observation"] += time.perf_counter() - t0
                
                t0 = time.perf_counter()
                fast_eval = run.fast_eval
                with torch.no_grad():
                    action, _, value = policy.get_action(
                        obs_ts, obs_sigs, act_sigs,
                        deterministic=True,
                        return_std=not fast_eval,
                        return_value=not fast_eval,
                    )
                profile_times["policy_get_action"] += time.perf_counter() - t0
                
                action_np = action.squeeze(0).cpu().numpy()
                
                t0 = time.perf_counter()
                if expert.has_expert:
                    expert_action = expert.get_expert_action(obs)
                    imitation_loss = float(((action_np - expert_action) ** 2).mean())
                    step_imitation_losses.append(imitation_loss)
                profile_times["get_expert_action"] += time.perf_counter() - t0
                
                t0 = time.perf_counter()
                next_obs, reward, terminated, truncated, info = expert.step(action_np)
                profile_times["expert_step"] += time.perf_counter() - t0
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                t0 = time.perf_counter()
                if not headless:
                    expert.env.render()
                    if sleep_per_step > 0:
                        time.sleep(sleep_per_step)
                profile_times["render"] += time.perf_counter() - t0
                
                t0 = time.perf_counter()
                parser.update(next_obs)
                profile_times["parser_update"] += time.perf_counter() - t0
                obs = next_obs
                
                if not profile_done:
                    profile_steps += 1
                
                if done:
                    if not profile_done and profile_steps > 0:
                        profile_done = True
                        total = sum(profile_times.values())
                        print("\n" + "-" * 50)
                        print("STEP PROFILE (first episode, ms per call)")
                        print("-" * 50)
                        for k, v in profile_times.items():
                            ms = 1000 * v / profile_steps
                            pct = 100 * v / total if total > 0 else 0
                            print(f"  {k:22s} {ms:8.2f} ms  ({pct:5.1f}%)")
                        print(f"  {'TOTAL':22s} {1000 * total / profile_steps:8.2f} ms  (100.0%)")
                        print("-" * 50)
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if step_imitation_losses:
                episode_imitation_losses.append(np.mean(step_imitation_losses))
            
            # Обновляем прогресс-бар
            postfix = {
                "reward": f"{episode_reward:.2f}",
                "length": episode_length,
            }
            if step_imitation_losses:
                postfix["im_loss"] = f"{np.mean(step_imitation_losses):.4f}"
            pbar.set_postfix(postfix)
    
    finally:
        expert.env.end_eval()
    
    # Итоговые метрики
    metrics = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
    }
    if episode_imitation_losses:
        metrics["imitation_loss"] = float(np.mean(episode_imitation_losses))
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Motions evaluated: {len(episode_rewards)}")
    print(f"  Mean reward:       {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
    print(f"  Mean length:       {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
    if "imitation_loss" in metrics:
        print(f"  Imitation loss:    {metrics['imitation_loss']:.6f}")
    print("=" * 60)
    
    return metrics


@hydra.main(config_path="../../cfg", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Entry point для валидации. Все параметры из cfg (run=validate, learning, device).
    """
    validate(cfg)


if __name__ == "__main__":
    main()
