import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class MultiAgentSuccessTrackingCallback(BaseCallback):
    """
    Callback robusto per ambienti multi-agent.
    Tiene traccia dei successi e della distanza finale di ciascun agente.
    Ripristina la stampa completa di info per monitoraggio training.
    """
    def __init__(self, target_radius=50.0, verbose=0, check_freq=1000):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.target_radius = target_radius

        # Dati per agente
        self.episode_successes = {}  # dict[agent] = lista successi
        self.episode_distances = {}  # dict[agent] = lista distanze
        self.n_episodes = {}         # dict[agent] = contatore episodi

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        if infos is None:
            infos = []

        # infos è lista di dict (un dict per ambiente)
        for info in infos:
            if isinstance(info, dict):
                # Cicla su agenti nel dict
                for agent, agent_info in info.items():
                    if isinstance(agent_info, dict) and 'distance_to_target' in agent_info:
                        self._update_agent(agent, agent_info)

        # Stampa periodica come prima
        if self.n_calls > 0 and self.n_calls % self.check_freq == 0:
            print(f"\n{'='*70}")
            print(f"📊 Progress at {self.n_calls:,} steps")
            print(f"{'='*70}")
            for agent in sorted(self.episode_successes):
                n_recent = len(self.episode_successes[agent])
                n_successes = sum(self.episode_successes[agent])
                success_rate = (n_successes / n_recent) * 100 if n_recent > 0 else 0.0
                avg_dist = np.mean(self.episode_distances[agent]) if n_recent > 0 else float('nan')
                print(f"Agent {agent} - Last {n_recent} episodes:")
                print(f"    ✓ Successes: {n_successes}/{n_recent} ({success_rate:.1f}%)")
                print(f"    Avg final distance: {avg_dist:.1f} m")
            print(f"{'='*70}\n")

        return True

    def _update_agent(self, agent, agent_info):
        """Aggiorna i dati di successo per un singolo agente."""
        if agent not in self.episode_successes:
            self.episode_successes[agent] = []
            self.episode_distances[agent] = []
            self.n_episodes[agent] = 0

        self.n_episodes[agent] += 1
        final_dist = agent_info.get('distance_to_target', np.inf)
        self.episode_distances[agent].append(final_dist)
        success = final_dist < self.target_radius
        self.episode_successes[agent].append(success)

        # mantieni solo le ultime 100 ep
        if len(self.episode_successes[agent]) > 100:
            self.episode_successes[agent].pop(0)
            self.episode_distances[agent].pop(0)