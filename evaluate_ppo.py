import os
import numpy as np
import imageio
from stable_baselines3 import PPO
from env_sailing_env import ImprovedSailingEnv


def _adapt_obs_for_model(obs_vector: np.ndarray, expected_dim: int) -> np.ndarray:
    """Match env observation size to model input size for backward compatibility."""
    current_dim = obs_vector.shape[0]
    if current_dim == expected_dim:
        return obs_vector
    if current_dim > expected_dim:
        return obs_vector[:expected_dim]
    return np.pad(obs_vector, (0, expected_dim - current_dim), mode='constant')

def create_video(model_path="models/sailing_ppo_improved", filename='videos/sailing_demo.mp4',
                 seed=None, wind_direction=None):
    """
    Genera video multi-agent mostrando per ogni barca:
    - distanza dal target
    - numero di step impiegati (se già arrivata)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print("=" * 70)
    print("SAILING VIDEO GENERATION (PettingZoo Multi-Agent)")
    print("=" * 70)

    # Carica modello
    model = PPO.load(model_path, device='cpu')
    env = ImprovedSailingEnv(render_mode='rgb_array')
    expected_obs_dim = int(model.observation_space.shape[0])
    env_obs_dim = int(env.observation_space(env.possible_agents[0]).shape[0])
    if expected_obs_dim != env_obs_dim:
        print(
            f"[WARN] Observation mismatch: model expects {expected_obs_dim}, "
            f"env provides {env_obs_dim}. Adapting observations for inference."
        )

    reset_opts = {}
    if wind_direction is not None:
        reset_opts['wind_direction'] = wind_direction

    obs, _ = env.reset(seed=seed, options=reset_opts if reset_opts else None)
    obs_array_dict = {
        agent: _adapt_obs_for_model(obs[agent], expected_obs_dim).reshape(1, -1)
        for agent in env.agents
    }

    frames = [env.render()]
    step = 0

    # Traccia step per cui ogni barca raggiunge il target
    step_reached = {agent: None for agent in env.possible_agents}
    dist_dict = {agent: 999 for agent in env.possible_agents}
    best_dist_dict = {agent: float('inf') for agent in env.possible_agents}

    while step < env.max_steps:
        actions = {
            agent: model.predict(obs_array_dict[agent], deterministic=True)[0][0]
            for agent in env.agents if agent in obs_array_dict
        }

        # Se tutte le barche hanno raggiunto il target, termina il loop
        if not actions:
            break

        obs, reward, terminated, truncated, info = env.step(actions)
        frames.append(env.render())
        step += 1

        # Aggiorna le osservazioni
        for agent in env.agents:
            if agent in obs:
                obs_array_dict[agent] = _adapt_obs_for_model(obs[agent], expected_obs_dim).reshape(1, -1)

        # Aggiorna distanze e registra gli step di arrivo individuali
        for agent in env.possible_agents:
            if agent in info:
                agent_info = info[agent]
                dist_dict[agent] = agent_info.get('distance_to_target', 999)
                best_dist = agent_info.get('best_distance', None)
                if isinstance(best_dist, (float, int)):
                    best_dist_dict[agent] = min(best_dist_dict[agent], float(best_dist))
                
                if step_reached[agent] is None and agent_info.get('finished_race', False):
                    step_reached[agent] = agent_info['steps_to_target']

    # Log finale
    for agent in env.possible_agents:
        sr = step_reached[agent]
        if sr is not None:
            print(f"   {agent} reached the target in {sr} steps!")
        else:
            best = best_dist_dict.get(agent, float('inf'))
            if np.isfinite(best):
                print(f"   {agent} did NOT reach the target. Best distance: {best:.1f} m")
            else:
                print(f"   {agent} did NOT reach the target. Best distance: ?")

    # Salva il video
    print(f"\nSaving video to {filename} ({len(frames)} frames)...")
    try:
        imageio.mimsave(filename, frames, fps=15)
        print(f"Video created: {filename}")
    except Exception as e:
        print(f"Error saving video: {e}")
        print("Assicurati di avere il plugin imageio-ffmpeg installato: pip install imageio[ffmpeg]")

    env.close()

def create_multi_video(model_path="models/sailing_ppo_realistic_until100", output_dir='videos'):
    """Genera 3 video di test della regata con posizioni di partenza casuali e barche multiple."""

    print("\n" + "=" * 70)
    print(f"  TEST FINALE: 3 video di regata")
    print("=" * 70)

    for i in range(1, 4):
        fname = os.path.join(output_dir, f'test_{i}_regata.mp4')
        print(f"\n--- Test {i}/3: Regata standard (Seed casuale {42 + i}) ---")
        create_video(
            model_path=model_path,
            filename=fname,
            seed=42 + i
        )

    print("\n" + "=" * 70)
    print(f"  3 video salvati in {output_dir}/")
    print("=" * 70)
