import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss

# Importazioni Locali
from env_sailing_env import ImprovedSailingEnv
from callbacks import SuccessTrackingCallback

def train_model(
    total_timesteps=500000,
    n_envs=4,
    model_path="models/sailing_ppo_improved",
    chunk_timesteps=None,
    max_chunks=None,
):
    """
    Funzione principale di training con supporto PARALLELISMO per PettingZoo tramite SuperSuit.
    n_envs: Numero di ambienti da eseguire in parallelo.
    """
    
    os.makedirs("./sailing_tensorboard", exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    print("="*70)
    print(f"🛥️  SAILING RL - TRAINING (Parallel Envs: {n_envs})")
    print("="*70)
    
    # 1. Creazione Ambiente Vettorizzato (Parallelismo)
    print(f"1. Creating {n_envs} parallel environments (PettingZoo via SuperSuit)...")
    
    env = ImprovedSailingEnv()
    agents_per_env = len(env.possible_agents)
    
    # Adatta l'ambiente PettingZoo a VecEnv Standard di SB3
    # Aggiungi black_death per permettere la rimozione di agenti in step diversi
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # Parallelizza l'ambiente (usa num_cpus=0 per evitare bug noti tra le ultime versioni di supersuit e SB3)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=n_envs, num_cpus=0, base_class='stable_baselines3')
    
    train_env = VecMonitor(env)
    
    # 2. Creazione del Modello PPO
    print("\n2. Creating PPO model...")
    # Keep rollout size stable and divisible by batch_size across different n_envs.
    rollout_steps_per_env = 256

    model = PPO(
        "MlpPolicy",
        train_env,
        device='cpu',
        learning_rate=2e-4,
        n_steps=rollout_steps_per_env,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log="./sailing_tensorboard/"
    )
    
    # 3. Setup Callback
    callback_check_freq = max(1, 10000 // max(1, n_envs * agents_per_env))
    callback = SuccessTrackingCallback(
        verbose=1,
        check_freq=callback_check_freq,
        window_size=200,
        success_window=100,
        expected_agents=["red_boat", "blue_boat"],
        stop_on_perfect_window=True,
    )
    
    # 4. Avvio Training
    print(f"\n4. Training for {total_timesteps} steps...")
    # Train in large chunks and stop only when callback reaches strict 100/100 recent success.
    if chunk_timesteps is None:
        chunk_timesteps = max(total_timesteps, 200_000)

    chunks_run = 0

    while not callback.goal_reached and (max_chunks is None or chunks_run < max_chunks):
        model.learn(total_timesteps=chunk_timesteps, callback=callback, reset_num_timesteps=False)
        chunks_run += 1
        if max_chunks is None:
            print(f"[Training] Completed chunk {chunks_run} (chunk_timesteps={chunk_timesteps})")
        else:
            print(f"[Training] Completed chunk {chunks_run}/{max_chunks} (chunk_timesteps={chunk_timesteps})")

    if callback.goal_reached:
        print("[Training] Stopped automatically after perfect recent success window.")
    else:
        print("[Training] Reached configured max_chunks before perfect recent success window.")
    
    # 5. Salvataggio Modello
    model.save(model_path)
    print(f"\n Model saved as '{model_path}'")
    
    return model
