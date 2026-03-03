import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv

# Importazioni Locali
from sailing_env import ImprovedSailingEnv
from callbacks import SuccessTrackingCallback

def train_model(total_timesteps=500000, n_envs=4, model_path="sailing_ppo_improved"):
    """
    Funzione principale di training con supporto PARALLELISMO.
    n_envs: Numero di ambienti da eseguire in parallelo.
    """
    
    os.makedirs("./sailing_tensorboard", exist_ok=True)
    
    print("="*70)
    print(f"🛥️  SAILING RL - TRAINING (Parallel Envs: {n_envs})")
    print("="*70)
    
    # 1. Creazione Ambiente Vettorizzato (Parallelismo)
    print(f"1. Creating {n_envs} parallel environments...")
    
    if n_envs > 1:
        train_env = make_vec_env(ImprovedSailingEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    else:
        train_env = make_vec_env(ImprovedSailingEnv, n_envs=n_envs)
    
    train_env = VecMonitor(train_env)
    
    # 2. Creazione del Modello PPO
    print("\n2. Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048 // n_envs,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.015,
        verbose=0,
        tensorboard_log="./sailing_tensorboard/"
    )
    
    # 3. Setup Callback
    callback = SuccessTrackingCallback(verbose=1, check_freq=10000 // n_envs)
    
    # 4. Avvio Training
    print(f"\n4. Training for {total_timesteps} steps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=False
    )
    
    # 5. Salvataggio Modello
    model.save(model_path)
    print(f"\n Model saved as '{model_path}'")
    
    return model
