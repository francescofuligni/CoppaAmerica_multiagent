#!/usr/bin/env python3
"""
Reinforcement Learning for Sailing (Multi-Agent)
Entry Point principale per gestire l'addestramento e la generazione video.
"""

import os
import argparse
from stable_baselines3 import PPO

# Moduli locali
from train_ppo import train_model
from evaluate_ppo import create_video, create_multi_video
from env_sailing_env import ImprovedSailingEnv


def is_model_compatible(model_path: str) -> bool:
    """Check if saved model spaces match current environment spaces."""
    model_zip = model_path + ".zip"
    if not os.path.exists(model_zip):
        return False

    env = ImprovedSailingEnv()
    try:
        model = PPO.load(model_path, device="cpu")
        sample_agent = env.possible_agents[0]
        env_obs_dim = int(env.observation_space(sample_agent).shape[0])
        env_act_dim = int(env.action_space(sample_agent).shape[0])
        model_obs_dim = int(model.observation_space.shape[0])
        model_act_dim = int(model.action_space.shape[0])

        if model_obs_dim != env_obs_dim or model_act_dim != env_act_dim:
            print(
                f"Model/environment mismatch: obs {model_obs_dim}!={env_obs_dim} "
                f"or act {model_act_dim}!={env_act_dim}. Retraining required."
            )
            return False
        return True
    except Exception as exc:
        print(f"Unable to validate model compatibility ({exc}). Retraining required.")
        return False
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gestore Addestramento Barca a Vela Multi-Agent"
    )
    parser.add_argument(
        "--train", action="store_true", help="Forza l'avvio del training"
    )
    parser.add_argument(
        "--test-multi",
        action="store_true",
        help="Genera 4 video con direzioni di vento differenti",
    )
    parser.add_argument(
        "--steps", type=int, default=1000000, help="Passi di training totali"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=14,
        help="Numero di ambienti/processi paralleli per training",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/sailing_ppo_improved",
        help="Nome file del modello",
    )
    parser.add_argument(
        "--video-file",
        type=str,
        default="videos/sailing_demo.mp4",
        help="Nome file video in output",
    )

    args = parser.parse_args()

    # 1 Training del modello se richiesto, mancante, o non compatibile con env
    # corrente
    model_ready = is_model_compatible(args.model_path)
    if args.train or not model_ready:
        print(f"Starting parallel training with {
                args.n_envs} environments for {
                args.steps} steps...")
        train_model(
            total_timesteps=args.steps, n_envs=args.n_envs, model_path=args.model_path
        )
    else:
        print(f"Model '{args.model_path}.zip' found. Skipping training.")
    # 2 Generazione video (multi-agent)
    if args.test_multi:
        print("\nGenerating multi-agent regatta videos...")
        create_multi_video(model_path=args.model_path)
    else:
        print(f"\nGenerating single multi-agent video: {args.video_file}")
        create_video(model_path=args.model_path, filename=args.video_file)
