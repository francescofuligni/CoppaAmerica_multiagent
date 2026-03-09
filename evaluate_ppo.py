import os
import numpy as np
import imageio
from stable_baselines3 import PPO

from sailing_env import ImprovedSailingEnv


def create_video(model_path="models/sailing_ppo_improved", filename='videos/sailing_demo.mp4',
                 seed=None, wind_direction=None):
    import os
    import numpy as np
    import imageio
    from stable_baselines3 import PPO
    from sailing_env import ImprovedSailingEnv

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print("=" * 70)
    print("SAILING VIDEO GENERATION (PettingZoo Multi-Agent)")
    print("=" * 70)

    model = PPO.load(model_path, device='cpu')
    env = ImprovedSailingEnv(render_mode='rgb_array')

    reset_opts = {}
    if wind_direction is not None:
        reset_opts['wind_direction'] = wind_direction

    obs, _ = env.reset(seed=seed, options=reset_opts if reset_opts else None)
    obs_array_dict = {agent: obs[agent].reshape(1, -1) for agent in env.agents}

    frames = [env.render()]
    step = 0

    # Salva gli step individuali per cui ogni barca raggiunge il target
    step_reached = {agent: None for agent in env.possible_agents}
    dist_dict = {agent: 999 for agent in env.possible_agents}

    while step < env.max_steps:
        actions = {}
        for agent, obs_array in obs_array_dict.items():
            # Predizione solo per agenti che non hanno ancora raggiunto il target
            if step_reached[agent] is None:
                action, _ = model.predict(obs_array, deterministic=True)
                actions[agent] = action[0]

        # Se tutti hanno già raggiunto il target, interrompi il loop
        if not actions:
            break

        obs, reward, terminated, truncated, info = env.step(actions)
        frames.append(env.render())
        step += 1

        # Aggiorna le osservazioni
        for agent in env.agents:
            if agent in obs:
                obs_array_dict[agent] = obs[agent].reshape(1, -1)

        # Aggiorna le distanze e registra gli step di arrivo individuali
        for agent in env.possible_agents:
            agent_info = info.get(agent, {})
            dist_dict[agent] = agent_info.get('distance_to_target', 999)
            if step_reached[agent] is None and dist_dict[agent] < env.target_radius:
                step_reached[agent] = step

        # Aggiorna il titolo di rendering con step individuali
        title_lines = []
        for agent in env.possible_agents:
            sr = step_reached[agent]
            if sr is not None:
                title_lines.append(f"{agent} reached: {sr} steps")
            else:
                title_lines.append(f"{agent} distance: {dist_dict[agent]:.1f}m")
        env.ax.set_title(" | ".join(title_lines), fontsize=10)

    # Log finale
    for agent in env.possible_agents:
        sr = step_reached[agent]
        if sr is not None:
            print(f"   {agent} reached the target in {sr} steps!")
        else:
            best = info.get(agent, {}).get('best_distance', '?')
            if isinstance(best, (float, int)):
                print(f"   {agent} did NOT reach the target. Best distance: {best:.1f}m")
            else:
                print(f"   {agent} did NOT reach the target. Best distance: {best}")

    # Salva il video
    print(f"\nSaving video to {filename} ({len(frames)} frames)...")
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
        ('Nord',  np.radians(90)),
        ('Est',   np.radians(0)),
        ('Sud',   np.radians(270)),
        ('Ovest', np.radians(180)),
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