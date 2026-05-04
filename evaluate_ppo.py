import os
import numpy as np
import imageio
from stable_baselines3 import PPO
from env.sailing_env import ImprovedSailingEnv
import supersuit as ss
from stable_baselines3.common.vec_env import VecMonitor, VecFrameStack
import yaml


def create_video(
    model_path="models/sailing_ppo_improved",
    filename="videos/sailing_demo.mp4",
    seed=None,
    wind_direction=None,
):
    """
    Genera video multi-agent mostrando per ogni barca:
    - distanza dal target
    - numero di step impiegati (se già arrivata)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print("=" * 70)
    print("SAILING VIDEO GENERATION (PettingZoo Multi-Agent Venv)")
    print("=" * 70)

    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}
    train_cfg = config.get("training", {})

    model = PPO.load(model_path, device="cpu")

    raw_env = ImprovedSailingEnv(render_mode="rgb_array")

    # Patch per far passare seed e wind_direction al reset sottostante in VecEnv
    orig_reset = raw_env.reset

    def custom_reset(*args, **kwargs):
        if wind_direction is not None:
            if "options" not in kwargs or kwargs["options"] is None:
                kwargs["options"] = {}
            kwargs["options"]["wind_direction"] = wind_direction
        if seed is not None:
            kwargs["seed"] = seed
        return orig_reset(*args, **kwargs)

    raw_env.reset = custom_reset

    n_stack = train_cfg.get("frame_stack", 4)
    raw_env = ss.frame_stack_v1(raw_env, n_stack)

    observations, infos = raw_env.reset()
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames = [raw_env.render()]
    
    step = 0
    while step < raw_env.unwrapped.max_steps:
        actions = {}
        for agent in raw_env.agents:
            obs = observations[agent]
            action, _ = model.predict(obs, deterministic=True)
            actions[agent] = action
        
        if not actions:
            break

        observations, rewards, terminations, truncations, infos = raw_env.step(actions)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frames.append(raw_env.render())
        
        step += 1

        # Interrompi solo quando le barche hanno finito (o sono state rimosse per timeout/fallimento)
        if all(terminations[a] or truncations[a] for a in terminations):
            break

    print(f"\nSaving video to {filename} ({len(frames)} frames)...")
    try:
        imageio.mimsave(filename, frames, fps=15)
        print(f"Video created: {filename}")
    except Exception as e:
        print(f"Error saving video: {e}")
        print(
            "Assicurati di avere il plugin imageio-ffmpeg installato: pip install imageio[ffmpeg]"
        )

    raw_env.close()


def create_multi_video(
    model_path="models/sailing_ppo_improved", output_dir="videos"
):
    """Genera 5 video di test della regata con posizioni di partenza casuali."""

    print("\n" + "=" * 70)
    print("  TEST FINALE: 5 video di regata")
    print("=" * 70)

    for i in range(1, 6):
        fname = os.path.join(output_dir, f"test_{i}.mp4")
        print(f"\n--- Test {i}/5: Regata standard ---")
        create_video(model_path=model_path, filename=fname, seed=42 + i)

    print("\n" + "=" * 70)
    print(f"  5 video salvati in {output_dir}/")
    print("=" * 70)
