#!/usr/bin/env python3
"""
Reinforcement Learning for Sailing (Multi-Agent)
Entry Point principale per gestire l'addestramento e la generazione video.
"""

import os
import argparse

# Moduli locali
from train_ppo import train_model
from evaluate_ppo import create_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gestore Addestramento Barca a Vela Multi-Agent")
    parser.add_argument("--train", action="store_true", help="Forza l'avvio del training")
    parser.add_argument("--steps", type=int, default=500000, help="Passi di training totali")
    parser.add_argument("--n-envs", type=int, default=12, help="Numero di ambienti paralleli per training")
    parser.add_argument("--model-path", type=str, default="models/sailing_ppo_improved", help="Nome file del modello")
    parser.add_argument("--video-file", type=str, default="videos/sailing_demo.mp4", help="Nome file video in output")
    
    args = parser.parse_args()
    
    # 1️⃣ Training del modello se richiesto o se non esiste il file
    if args.train or not os.path.exists(args.model_path + ".zip"):
        print(f"Starting parallel training with {args.n_envs} environments for {args.steps} steps...")
        train_model(total_timesteps=args.steps, n_envs=args.n_envs, model_path=args.model_path)
    else:
        print(f"Model '{args.model_path}.zip' found. Skipping training.")
    
    # 2️⃣ Generazione di un singolo video con tutte le barche (multi-agent)
    print(f"\nGenerating single multi-agent video: {args.video_file}")
    create_video(model_path=args.model_path, filename=args.video_file)