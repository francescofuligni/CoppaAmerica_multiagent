import imageio
from stable_baselines3 import PPO

# Importazioni locali
from sailing_env import ImprovedSailingEnv

def create_video(model_path="sailing_ppo_improved", filename='sailing_demo.mp4'):
    """Genera video (singolo thread per la visualizzazione grafica)"""
    
    print("="*70)
    print("🎬 SAILING VIDEO GENERATION")
    print("="*70)
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        return

    # Usiamo un singolo ambiente per il video con render_mode attivo
    env = ImprovedSailingEnv(render_mode='rgb_array')
    frames = []
    
    obs, _ = env.reset()
    frames.append(env._render_frame())
    
    done = False
    step = 0
    info = {'distance_to_target': 999}
    
    while not done and step < 250:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env._render_frame())
        done = terminated or truncated
        step += 1
        
        if terminated and info['distance_to_target'] < 50.0:
            print(f"   ✓ Target reached in {step} steps!")
            for _ in range(20):
                frames.append(env._render_frame())
    
    print(f"\n3. Saving video to {filename} ({len(frames)} frames)...")
    try:
        imageio.mimsave(filename, frames, fps=15)
        print(f"✓ Video created: {filename}")
    except Exception as e:
         print(f"⚠️ Error saving video: {e}")
         print("Assicurati di avere il plugin imageio-ffmpeg installato: pip install imageio[ffmpeg]")
         
    env.close()
    
