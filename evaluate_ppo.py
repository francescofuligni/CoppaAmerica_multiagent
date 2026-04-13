import os
import numpy as np
import imageio
from stable_baselines3 import PPO
from env_sailing_env import ImprovedSailingEnv
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

    # Carica modello
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

    venv = ss.pettingzoo_env_to_vec_env_v1(raw_env)
    venv = ss.concat_vec_envs_v1(venv, 1, num_cpus=0, base_class="stable_baselines3")
    venv = VecMonitor(venv)
    n_stack = train_cfg.get("frame_stack", 4)
    venv = VecFrameStack(venv, n_stack=n_stack)

    obs = venv.reset()
    frames = [raw_env.render()]
    step = 0

    # Non possiamo tracciare le info individuali in un loop vettoriale compatto 
    # facilmente come prima se l'ambiente si resetta da solo, 
    # ma possiamo generare il video fluido.
    while step < raw_env.max_steps:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = venv.step(actions)
        frames.append(raw_env.render())
        step += 1
        
        # In multi-agent sb3 con concat_vec_envs 2 agents = 2 envs array.
        # dones è un array booleano [False, False]
        if all(dones):
            break

    # Salva il video
    print(f"\nSaving video to {filename} ({len(frames)} frames)...")
    try:
        imageio.mimsave(filename, frames, fps=15)
        print(f"Video created: {filename}")
    except Exception as e:
        print(f"Error saving video: {e}")
        print(
            "Assicurati di avere il plugin imageio-ffmpeg installato: pip install imageio[ffmpeg]"
        )

    venv.close()


def create_multi_video(
    model_path="models/sailing_ppo_realistic_until100", output_dir="videos"
):
    """Genera 3 video di test della regata con posizioni di partenza casuali e barche multiple."""

    print("\n" + "=" * 70)
    print("  TEST FINALE: 5 video di regata")
    print("=" * 70)

    for i in range(1, 6):
        fname = os.path.join(output_dir, f"test_{i}.mp4")
        print(f"\n--- Test {i}/5: Regata standard (Seed casuale {42 + i}) ---")
        create_video(model_path=model_path, filename=fname, seed=42 + i)

    print("\n" + "=" * 70)
    print(f"  5 video salvati in {output_dir}/")
    print("=" * 70)
