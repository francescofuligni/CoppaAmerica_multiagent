import os
import numpy as np
import imageio
from stable_baselines3 import PPO
from env_sailing_env import ImprovedSailingEnv


def _adapt_obs_for_model(obs_vector: np.ndarray, expected_dim: int) -> np.ndarray:
    """Adatta la shape osservazione per compatibilita' con modelli salvati."""
    current_dim = obs_vector.shape[0]
    if current_dim == expected_dim:
        return obs_vector
    if current_dim > expected_dim:
        return obs_vector[:expected_dim]
    return np.pad(obs_vector, (0, expected_dim - current_dim), mode="constant")


def create_video(
    model_path="models/sailing_ppo_improved",
    filename="videos/sailing_demo.mp4",
    seed=None,
    wind_direction=None,
):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print("=" * 70)
    print("SAILING VIDEO GENERATION (PettingZoo Multi-Agent Dynamic Env)")
    print("=" * 70)

    model = PPO.load(model_path, device="cpu")
    env = ImprovedSailingEnv(render_mode="rgb_array")

    expected_obs_dim = int(model.observation_space.shape[0])
    env_obs_dim = int(env.observation_space(env.possible_agents[0]).shape[0])
    if expected_obs_dim != env_obs_dim:
        print(
            f"[WARN] Observation mismatch: model expects {expected_obs_dim}, "
            f"env provides {env_obs_dim}. Adapting observations for inference."
        )

    reset_opts = {}
    if wind_direction is not None:
        reset_opts["wind_direction"] = wind_direction

    obs, _ = env.reset(seed=seed, options=reset_opts if reset_opts else None)
    obs_array_dict = {
        agent: _adapt_obs_for_model(obs[agent], expected_obs_dim).reshape(1, -1)
        for agent in env.agents
    }

    frames = [env.render()]
    step = 0

    step_reached = {agent: None for agent in env.possible_agents}
    best_dist_dict = {agent: float("inf") for agent in env.possible_agents}

    while step < env.max_steps:
        actions = {}
        for agent in env.agents:
            if agent in obs_array_dict:
                action, _ = model.predict(obs_array_dict[agent], deterministic=True)
                actions[agent] = action[0]

        if not actions:
            break

        obs, reward, terminated, truncated, info = env.step(actions)
        frames.append(env.render())
        step += 1

        for agent in env.agents:
            if agent in obs:
                obs_array_dict[agent] = _adapt_obs_for_model(
                    obs[agent], expected_obs_dim
                ).reshape(1, -1)

        for agent in env.possible_agents:
            if agent in info:
                agent_info = info[agent]
                best_dist = agent_info.get("best_distance", None)
                if isinstance(best_dist, (float, int)):
                    best_dist_dict[agent] = min(best_dist_dict[agent], float(best_dist))

                if step_reached[agent] is None and agent_info.get("finished_race", False):
                    step_reached[agent] = agent_info.get("steps_to_target")

    for agent in env.possible_agents:
        sr = step_reached[agent]
        if sr is not None:
            print(f"   {agent} reached the target in {sr} steps!")
        else:
            best = best_dist_dict.get(agent, float("inf"))
            if np.isfinite(best):
                print(f"   {agent} did NOT reach the target. Best distance: {best:.1f} m")
            else:
                print(f"   {agent} did NOT reach the target. Best distance: ?")

    print(f"\nSaving video to {filename} ({len(frames)} frames)...")
    try:
        imageio.mimsave(filename, frames, fps=15)
        print(f"Video created: {filename}")
    except Exception as e:
        print(f"Error saving video: {e}")
        print("Assicurati di avere imageio-ffmpeg: pip install imageio[ffmpeg]")

    env.close()


def create_multi_video(model_path="models/sailing_ppo_improved", output_dir="videos"):
    print("\n" + "=" * 70)
    print("  TEST FINALE: 3 video di regata (dynamic env)")
    print("=" * 70)

    for i in range(1, 4):
        fname = os.path.join(output_dir, f"test_dynamic_{i}.mp4")
        print(f"\n--- Test {i}/3: Regata standard (Seed casuale {42 + i}) ---")
        create_video(model_path=model_path, filename=fname, seed=42 + i)

    print("\n" + "=" * 70)
    print(f"  3 video salvati in {output_dir}/")
    print("=" * 70)
