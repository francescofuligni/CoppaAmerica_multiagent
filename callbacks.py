import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SuccessTrackingCallback(BaseCallback):
    """
    Callback robusta per ambienti multi-agente e vettoriali (parallelizzati con VecEnv).
    Tiene traccia di successi e distanze finali di ciascun agente.
    """
    def __init__(self, verbose=1, check_freq=1000, target_radius=50.0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.target_radius = target_radius

        # Dati: agent -> lista successi e distanze
        self.episode_successes = {}  # dict[agent] = lista successi
        self.episode_distances = {}  # dict[agent] = lista distanze
        self.n_episodes = {}         # dict[agent] = totale episodi

    def _on_step(self) -> bool:
        # In vectorized envs, self.locals['infos'] è una lista con un dict per ambiente
        infos = self.locals.get("infos", [])

        for env_info in infos:
            if isinstance(env_info, dict):
                # Cicla sugli agenti dentro ogni ambiente
                for agent, agent_info in env_info.items():
                    if not isinstance(agent_info, dict):
                        continue
                    if 'distance_to_target' not in agent_info:
                        continue

                    if agent not in self.episode_successes:
                        self.episode_successes[agent] = []
                        self.episode_distances[agent] = []
                        self.n_episodes[agent] = 0

                    final_dist = agent_info['distance_to_target']
                    success = final_dist < self.target_radius

                    self.episode_successes[agent].append(success)
                    self.episode_distances[agent].append(final_dist)
                    self.n_episodes[agent] += 1

                    # Mantieni solo ultime 100 ep
                    if len(self.episode_successes[agent]) > 100:
                        self.episode_successes[agent].pop(0)
                        self.episode_distances[agent].pop(0)

        # Stampa periodica
        if self.n_calls > 0 and self.n_calls % self.check_freq == 0:
            print(f"\n{'='*70}")
            print(f"📊 Progress at {self.n_calls:,} steps")
            print(f"{'='*70}")
            for agent in sorted(self.episode_successes):
                n_recent = len(self.episode_successes[agent])
                if n_recent > 0:
                    n_successes = sum(self.episode_successes[agent])
                    success_rate = (n_successes / n_recent) * 100
                    avg_dist = np.mean(self.episode_distances[agent])
                    print(f"{agent} - Last {n_recent} episodes:")
                    print(f"   ✓ Successes: {n_successes}/{n_recent} ({success_rate:.1f}%)")
                    print(f"   Avg final distance: {avg_dist:.1f} m")
                else:
                    print(f"{agent} - No episodes completed yet")
            print(f"{'='*70}\n")

        return True