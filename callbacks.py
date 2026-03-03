import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SuccessTrackingCallback(BaseCallback):
    """
    Callback aggiornata per supportare ambienti vettoriali (Multi-processing).
    """
    def __init__(self, verbose=0, check_freq=1000):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_successes = []
        self.episode_distances = []
        self.n_episodes = 0
        self.target_radius = 50.0

    def _on_step(self) -> bool:
        if 'infos' in self.locals:
            for i, info in enumerate(self.locals['infos']):
                if 'episode' in info:
                    self.n_episodes += 1
                    
                    final_dist = info.get('distance_to_target', 999)
                    
                    self.episode_distances.append(final_dist)
                    success = final_dist < self.target_radius
                    self.episode_successes.append(success)
                    
                    if len(self.episode_successes) > 100:
                        self.episode_successes.pop(0)
                        self.episode_distances.pop(0)

        if self.n_calls > 0 and self.n_calls % self.check_freq == 0:
            print(f"\n{'='*70}")
            print(f"📊 Progress at {self.n_calls:,} steps (Total Episodes: {self.n_episodes})")
            print(f"{'='*70}")
            
            if len(self.episode_successes) > 0:
                n_recent = len(self.episode_successes)
                n_successes = sum(self.episode_successes)
                success_rate = (n_successes / n_recent) * 100
                
                print(f"  Last {n_recent} episodes:")
                print(f"    ✓ Successes: {n_successes}/{n_recent} ({success_rate:.1f}%)")
                
                avg_dist = np.mean(self.episode_distances)
                print(f"    Avg final distance: {avg_dist:.1f}m")
            else:
                print(f"  ⏳ No episodes completed yet...")
                
            print(f"{'='*70}\n")
            
        return True
