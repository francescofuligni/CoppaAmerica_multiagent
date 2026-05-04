#!/usr/bin/env python3
"""Entry point per addestramento e generazione video."""

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
    """Verifica se gli spazi del modello salvato corrispondono a quelli dell'ambiente corrente."""
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
        description="Multi-Agent Sailing Training Manager"
    )
    parser.add_argument(
        "--train-new",
        action="store_true",
        help="Create a NEW model, resetting old checkpoints",
    )
    parser.add_argument(
        "--train-resume",
        action="store_true",
        help="Resume training from the LAST existing model",
    )
    parser.add_argument(
        "--test-multi",
        action="store_true",
        help="Generate 5 videos with different random seeds",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=run_cfg.get("steps", 1000000),
        help="Total training steps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=run_cfg.get("n_envs", 14),
        help="Number of parallel environments/processes for training",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=run_cfg.get("model_name", "sailing_model"),
        help="BASE name of the model",
    )
    parser.add_argument(
        "--video-file",
        type=str,
        default=run_cfg.get("video_file", "videos/sailing_demo.mp4"),
        help="Output video filename for single test",
    )

    args = parser.parse_args()

    if args.train_new or args.train_resume:
        model_path = resolve_model_path(args.model_name, create_new=args.train_new)
        print(f"\n[System] Resolved Model Target: {model_path}.zip")

        if not args.train_new:
            model_ready = is_model_compatible(model_path)
            if not model_ready:
                print(
                    "ERROR: Unable to resume. Incompatible or missing model."
                )
                exit(1)

        print(
            f"Starting parallel training ({args.n_envs} envs) for {args.steps} steps..."
        )

        try:
            train_model(
                total_timesteps=args.steps, n_envs=args.n_envs, model_path=model_path
            )
        except KeyboardInterrupt:
            print(
                "\n[!] Training manually interrupted (CTRL+C). Launching visual test of saved model..."
            )
        finally:
            print(
                f"\n[System] Auto-generating post-training video using {model_path}.zip..."
            )
            if args.test_multi:
                create_multi_video(model_path=model_path)
            else:
                create_video(model_path=model_path, filename=args.video_file)
    else:
        model_path = resolve_model_path(args.model_name, create_new=False)
        print(f"\n[System] Resolved Model Target for Test: {model_path}.zip")
        if args.test_multi:
            print("\nGenerating videos...")
            create_multi_video(model_path=model_path)
        else:
            print(f"\nGenerating single multi-agent video: {args.video_file}...")
            create_video(model_path=model_path, filename=args.video_file)
