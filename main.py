#!/usr/bin/env python3
"""
Reinforcement Learning for Sailing (Multi-Agent)
Entry Point principale per gestire l'addestramento e la generazione video.
"""

import os
import argparse
import warnings
import glob
import re
import yaml

warnings.filterwarnings(
    "ignore", category=UserWarning, module="stable_baselines3.common.vec_env.base_vec_env"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="supersuit.vector.sb3_vector_wrapper"
)
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")
from stable_baselines3 import PPO

from train_ppo import train_model
from evaluate_ppo import create_video, create_multi_video
from env.sailing_env import ImprovedSailingEnv


def resolve_model_path(base_name: str, create_new: bool) -> str:
    """Risolve il path corretto per supportare il versionamento automatico."""
    os.makedirs("models", exist_ok=True)
    pattern = os.path.join("models", f"{base_name}*.zip")
    files = glob.glob(pattern)

    max_version = 0
    regex = re.compile(rf"^{re.escape(base_name)}(?:_(\d+))?\.zip$")

    for f in files:
        basename = os.path.basename(f)
        match = regex.match(basename)
        if match:
            ver_str = match.group(1)
            ver = int(ver_str) if ver_str else 1
            if ver > max_version:
                max_version = ver

    if create_new:
        next_ver = max_version + 1
        suffix = f"_{next_ver}" if next_ver > 1 else ""
        return os.path.join("models", f"{base_name}{suffix}")
    else:
        if max_version == 0:
            return os.path.join("models", base_name)
        suffix = f"_{max_version}" if max_version > 1 else ""
        return os.path.join("models", f"{base_name}{suffix}")


def is_model_compatible(model_path: str) -> bool:
    """Check if saved model spaces match current environment spaces."""
    model_zip = model_path + ".zip"
    if not os.path.exists(model_zip):
        return False

    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}
    frame_stack = config.get("training", {}).get("frame_stack", 1)

    env = ImprovedSailingEnv()
    try:
        model = PPO.load(model_path, device="cpu")
        sample_agent = env.possible_agents[0]
        env_obs_dim = int(env.observation_space(sample_agent).shape[0]) * frame_stack
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
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}
        
    run_cfg = config.get("run", {})

    parser = argparse.ArgumentParser(
        description="Gestore Addestramento Barca a Vela Multi-Agent"
    )
    parser.add_argument(
        "--train-new",
        action="store_true",
        help="Crea un NUOVO modello azzerando i vecchi checkpoint",
    )
    parser.add_argument(
        "--train-resume",
        action="store_true",
        help="Riprende l'addestramento dall'ULTIMO modello esistente",
    )
    parser.add_argument(
        "--test-multi",
        action="store_true",
        help="Genera 5 video con differenti seed casuali",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=run_cfg.get("steps", 1000000),
        help="Passi di training totali",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=run_cfg.get("n_envs", 14),
        help="Numero di ambienti/processi paralleli per training",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=run_cfg.get("model_name", "sailing_model"),
        help="Nome BASE del modello",
    )
    parser.add_argument(
        "--video-file",
        type=str,
        default=run_cfg.get("video_file", "videos/sailing_demo.mp4"),
        help="Nome file video in output per il test singolo",
    )

    args = parser.parse_args()

    if args.train_new or args.train_resume:
        model_path = resolve_model_path(args.model_name, create_new=args.train_new)
        print(f"\n[Sistema] Target Modello Risolto: {model_path}.zip")

        if not args.train_new:
            model_ready = is_model_compatible(model_path)
            if not model_ready:
                print(
                    "ERRORE: Impossibile riprendere. Modello incompatibile o non trovato."
                )
                exit(1)

        print(
            f"Avvio addestramento parallelo ({args.n_envs} envs) per {args.steps} steps..."
        )

        try:
            train_model(
                total_timesteps=args.steps, n_envs=args.n_envs, model_path=model_path
            )
        except KeyboardInterrupt:
            print(
                "\n[!] Addestramento interrotto manualmente (CTRL+C). Avvio test visivo del modello salvato..."
            )
        finally:
            print(
                f"\n[Sistema] Generazione automatica video post-addestramento usando {model_path}.zip..."
            )
            if args.test_multi:
                create_multi_video(model_path=model_path)
            else:
                create_video(model_path=model_path, filename=args.video_file)
    else:
        model_path = resolve_model_path(args.model_name, create_new=False)
        print(f"\n[Sistema] Target Modello Risolto per Test: {model_path}.zip")
        if args.test_multi:
            print("\nGenerating multi-agent regatta videos...")
            create_multi_video(model_path=model_path)
        else:
            print(f"\nGenerating single multi-agent video: {args.video_file}...")
            create_video(model_path=model_path, filename=args.video_file)
