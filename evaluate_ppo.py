import os
import numpy as np
import imageio
from stable_baselines3 import PPO

# Importazioni locali
from sailing_env import ImprovedSailingEnv


def create_video(model_path="models/sailing_ppo_improved", filename='videos/sailing_demo.mp4',
                 seed=None, wind_direction=None):
    """Genera video renderizzando direttamente dall'ambiente PettingZoo (senza SuperSuit)."""

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print("=" * 70)
    print("SAILING VIDEO GENERATION (PettingZoo)")
    print("=" * 70)

    try:
        model = PPO.load(model_path, device='cpu')
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        return

    env = ImprovedSailingEnv(render_mode='rgb_array')
    agent = 'boat_0'

    reset_opts = {}
    if wind_direction is not None:
        reset_opts['wind_direction'] = wind_direction
    obs, _ = env.reset(seed=seed, options=reset_opts if reset_opts else None)
    obs_array = obs[agent].reshape(1, -1)

    frames = [env.render()]

    done = False
    step = 0
    dist = 999

    while not done and step < env.max_steps:
        action, _ = model.predict(obs_array, deterministic=True)
        actions = {agent: action[0]}
        obs, reward, terminated, truncated, info = env.step(actions)
        frames.append(env.render())

        done = terminated.get(agent, False) or truncated.get(agent, False)
        step += 1

        if agent in obs:
            obs_array = obs[agent].reshape(1, -1)

        dist = info.get(agent, {}).get('distance_to_target', 999)

        if done and dist < env.target_radius:
            print(f"   Target reached in {step} steps!")
            # Pausa sul frame finale (stato reale, nessun auto-reset)
            for _ in range(20):
                frames.append(frames[-1])

    if not done or dist >= env.target_radius:
        best = info.get(agent, {}).get('best_distance', '?')
        print(f"   Target NOT reached. Best distance: {best}m")

    print(f"\n3. Saving video to {filename} ({len(frames)} frames)...")
    try:
        imageio.mimsave(filename, frames, fps=15)
        print(f"Video created: {filename}")
    except Exception as e:
        print(f"Error saving video: {e}")
        print("Assicurati di avere il plugin imageio-ffmpeg installato: pip install imageio[ffmpeg]")

    env.close()


def create_multi_video(model_path="models/sailing_ppo_improved", output_dir='videos'):
    """Genera 4 video con direzioni di vento diverse (N, E, S, W)."""
    wind_configs = [
        ('Nord',  np.radians(90)),   # vento da Sud verso Nord (math: 90°)
        ('Est',   np.radians(0)),    # vento da Ovest verso Est (math: 0°)
        ('Sud',   np.radians(270)),  # vento da Nord verso Sud (math: 270°)
        ('Ovest', np.radians(180)),  # vento da Est verso Ovest (math: 180°)
    ]

    print("\n" + "=" * 70)
    print(f"  TEST FINALE: 4 video con vento diverso")
    print("=" * 70)

    for i, (label, wind_dir) in enumerate(wind_configs, 1):
        compass_deg = (90 - np.degrees(wind_dir)) % 360
        fname = os.path.join(output_dir, f'test_{i}_vento_{label}.mp4')
        print(f"\n--- Test {i}/4: Vento verso {label} ({compass_deg:.0f} bussola) ---")
        create_video(
            model_path=model_path,
            filename=fname,
            seed=42 + i,
            wind_direction=wind_dir,
        )

    print("\n" + "=" * 70)
    print(f"  4 video salvati in {output_dir}/")
    print("=" * 70)
