import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss

# Importazioni Locali (nuovo env modulare)
from env_sailing_env import ImprovedSailingEnv
from callbacks import SuccessTrackingCallback


def train_model(
    total_timesteps=500000,
    n_envs=4,
    model_path="models/sailing_ppo_improved",
    chunk_timesteps=None,
    max_chunks=None,
    ppo_preset="safe_optimized",
):
    """
    Training PPO su env modulare (env_sailing_env) mantenendo API e preset correnti.
    """

    os.makedirs("./sailing_tensorboard", exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print("=" * 70)
    print(f"SAILING RL - TRAINING DYNAMIC (Parallel Envs: {n_envs})")
    print("=" * 70)

    print(f"1. Creating {n_envs} parallel environments (PettingZoo via SuperSuit)...")

    env = ImprovedSailingEnv()
    agents_per_env = len(env.possible_agents)

    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=n_envs,
        num_cpus=0,
        base_class="stable_baselines3",
    )

    train_env = VecMonitor(env)

    print("\n2. Creating PPO model...")

    ppo_presets = {
        "legacy": {
            "learning_rate": 2e-4,
            "n_steps": 256,
            "batch_size": 256,
            "gamma": 0.99,
        },
        "safe_optimized": {
            "learning_rate": 3e-4,
            "n_steps": 1024,
            "batch_size": 512,
            "gamma": 0.995,
        },
    }

    if ppo_preset not in ppo_presets:
        valid = ", ".join(sorted(ppo_presets))
        raise ValueError(f"Unknown ppo_preset '{ppo_preset}'. Valid values: {valid}")

    ppo_cfg = ppo_presets[ppo_preset]
    print(f"   Using PPO preset: {ppo_preset}")

    model = PPO(
        "MlpPolicy",
        train_env,
        device="cpu",
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=10,
        gamma=ppo_cfg["gamma"],
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log="./sailing_tensorboard/",
    )

    callback_check_freq = max(1, 10000 // max(1, n_envs * agents_per_env))
    callback = SuccessTrackingCallback(
        verbose=1,
        check_freq=callback_check_freq,
        window_size=200,
        success_window=100,
        expected_agents=["red_boat", "blue_boat"],
        stop_on_perfect_window=True,
    )

    print(f"\n4. Training for {total_timesteps} steps...")
    if chunk_timesteps is None:
        chunk_timesteps = max(total_timesteps, 200_000)

    chunks_run = 0
    try:
        while not callback.goal_reached and (max_chunks is None or chunks_run < max_chunks):
            model.learn(
                total_timesteps=chunk_timesteps,
                callback=callback,
                reset_num_timesteps=False,
            )
            chunks_run += 1
            if max_chunks is None:
                print(f"[Training] Completed chunk {chunks_run} (chunk_timesteps={chunk_timesteps})")
            else:
                print(
                    f"[Training] Completed chunk {chunks_run}/{max_chunks} "
                    f"(chunk_timesteps={chunk_timesteps})"
                )
    except KeyboardInterrupt:
        print(
            "\n\n[Training] Interrotto manualmente dall'utente (Ctrl+C). Procedo al salvataggio finale del modello..."
        )

    if callback.goal_reached:
        print("[Training] Stopped automatically after perfect recent success window.")
    else:
        print("[Training] Training ended (either manually interrupted or max_chunks reached).")

    model.save(model_path)
    print(f"\nModel saved as '{model_path}'")

    return model
