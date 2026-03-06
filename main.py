#!/usr/bin/env python3
"""
Reinforcement Learning for Sailing (Refactored)
Entry Point principale per gestire l'addestramento e i video da terminale.
"""

import os
import argparse

# Importazioni dai nostri moduli separati
from train_ppo import train_model
from evaluate_ppo import create_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gestore Addestramento Barca a Vela")
    parser.add_argument("--train", action="store_true", help="Forza l'avvio del training")
    parser.add_argument("--steps", type=int, default=500000, help="Passi di training totali")
    parser.add_argument("--n-envs", type=int, default=12, help="Numero di ambienti/processi paralleli")
    parser.add_argument("--model-path", type=str, default="models/sailing_ppo_improved", help="Nome file del modello")
    parser.add_argument("--video-file", type=str, default="videos/sailing_demo.mp4", help="Nome file video in output")
    
    args = parser.parse_args()
    
    # Eseguiamo il train se richiesto "--train" OPPURE se il file del modello non esiste ancora
    if args.train or not os.path.exists(args.model_path + ".zip"):
        print(f"Starting parallel training with {args.n_envs} environments for {args.steps} steps...")
        train_model(total_timesteps=args.steps, n_envs=args.n_envs, model_path=args.model_path)
    else:
        print(f"Model '{args.model_path}.zip' found. Skipping training.")
        
    # Crea sempre il video alla fine
    create_video(model_path=args.model_path, filename=args.video_file)
