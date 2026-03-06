import os
import imageio
from stable_baselines3 import PPO
import supersuit as ss

# Importazioni locali
from sailing_env import ImprovedSailingEnv

def create_video(model_path="models/sailing_ppo_improved", filename='videos/sailing_demo.mp4'):
    """Genera video (singolo thread per la visualizzazione grafica)"""
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    print("="*70)
    print("🎬 SAILING VIDEO GENERATION (PettingZoo)")
    print("="*70)
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        return

    # Usiamo un singolo ambiente per il video con render_mode attivo
    pz_env = ImprovedSailingEnv(render_mode='rgb_array')
    env = ss.pettingzoo_env_to_vec_env_v1(pz_env)
    # Importante: num_cpus=0 altrimenti supersuit crea il processo in background e pz_env originale non viene aggiornato!
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=0, base_class='stable_baselines3')
    
    frames = []
    
    obs = env.reset()
    frames.append(env.render())
    
    done = False
    step = 0
    info_dist = 999
    
    while not done and step < 250:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, info = env.step(action)
        frames.append(env.render())
        done = terminated[0]
        step += 1
        
        if len(info) > 0 and 'distance_to_target' in info[0]:
             info_dist = info[0]['distance_to_target']
        
        if done and info_dist < 50.0:
            print(f"   ✓ Target reached in {step} steps!")
            for _ in range(20):
                frames.append(env.render())
    
    print(f"\n3. Saving video to {filename} ({len(frames)} frames)...")
    try:
        imageio.mimsave(filename, frames, fps=15)
        print(f"✓ Video created: {filename}")
    except Exception as e:
         print(f"⚠️ Error saving video: {e}")
         print("Assicurati di avere il plugin imageio-ffmpeg installato: pip install imageio[ffmpeg]")
         
    env.close()
