import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SuccessTrackingCallback(BaseCallback):
    """
    Callback robusta per ambienti multi-agente e vettoriali (parallelizzati con VecEnv).
    Tiene traccia di successi e distanze finali di ciascun agente.
    """
    def __init__(
        self,
        verbose=1,
        check_freq=1000,
        target_radius=50.0,
        window_size=200,
        success_window=100,
        expected_agents=None,
        stop_on_perfect_window=True,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.target_radius = target_radius
        self.window_size = window_size
        self.success_window = success_window
        self.expected_agents = list(expected_agents) if expected_agents is not None else None
        self.stop_on_perfect_window = stop_on_perfect_window
        self.goal_reached = False

        # Dati: agent -> lista successi e distanze
        self.episode_successes = {}  # dict[agent] = lista successi
        self.episode_distances = {}  # dict[agent] = lista distanze
        self.n_episodes = {}         # dict[agent] = totale episodi

        # Metriche continue (rolling window)
        self.trim_eff_window = {}
        self.trim_error_window = {}
        self.vmg_window = {}
        self.speed_window = {}

    def _ensure_agent_buffers(self, agent: str) -> None:
        if agent in self.episode_successes:
            return
        self.episode_successes[agent] = []
        self.episode_distances[agent] = []
        self.n_episodes[agent] = 0
        self.trim_eff_window[agent] = []
        self.trim_error_window[agent] = []
        self.vmg_window[agent] = []
        self.speed_window[agent] = []

    def _append_rolling(self, buffer, value: float) -> None:
        buffer.append(value)
        if len(buffer) > self.window_size:
            buffer.pop(0)

    def _consume_agent_info(self, agent: str, agent_info: dict) -> None:
        """Consume metrics for one agent sample (supports both step and terminal records)."""
        self._ensure_agent_buffers(agent)

        if 'trim_efficiency' in agent_info:
            self._append_rolling(self.trim_eff_window[agent], float(agent_info['trim_efficiency']))
        if 'trim_error' in agent_info:
            self._append_rolling(self.trim_error_window[agent], float(agent_info['trim_error']))
        if 'vmg' in agent_info:
            self._append_rolling(self.vmg_window[agent], float(agent_info['vmg']))
        if 'speed' in agent_info:
            self._append_rolling(self.speed_window[agent], float(agent_info['speed']))

        # Conta episodio solo in evento finale.
        is_terminal = bool(agent_info.get('terminated', False) or agent_info.get('truncated', False))
        if not is_terminal:
            return

        final_dist = float(agent_info['distance_to_target'])
        if 'finished_race' in agent_info:
            success = bool(agent_info['finished_race'])
        else:
            success = final_dist < self.target_radius

        self.episode_successes[agent].append(success)
        self.episode_distances[agent].append(final_dist)
        self.n_episodes[agent] += 1

        # Mantieni solo ultime N ep (N = max finestra metriche e finestra obiettivo)
        keep_n = max(100, self.success_window)
        if len(self.episode_successes[agent]) > keep_n:
            self.episode_successes[agent].pop(0)
            self.episode_distances[agent].pop(0)

    def _has_perfect_recent_window(self) -> bool:
        if self.expected_agents is not None:
            agents = self.expected_agents
        else:
            agents = sorted(self.episode_successes.keys())

        if not agents:
            return False

        for agent in agents:
            successes = self.episode_successes.get(agent, [])
            if len(successes) < self.success_window:
                return False
            recent = successes[-self.success_window:]
            if sum(recent) != self.success_window:
                return False

        return True

    def _on_step(self) -> bool:
        # In vectorized envs, self.locals['infos'] è una lista con un dict per ambiente
        infos = self.locals.get("infos", [])

        for env_info in infos:
            if not isinstance(env_info, dict):
                continue

            # Formato appiattito (VecEnv after PettingZoo conversion): singolo agent_info.
            if 'distance_to_target' in env_info:
                agent = str(env_info.get('agent', 'agent_0'))
                self._consume_agent_info(agent, env_info)
                continue

            # Formato nidificato: {agent_name: agent_info}
            for agent, agent_info in env_info.items():
                if not isinstance(agent_info, dict):
                    continue
                if 'distance_to_target' not in agent_info:
                    continue
                self._consume_agent_info(str(agent), agent_info)

        if self.stop_on_perfect_window and self._has_perfect_recent_window():
            self.goal_reached = True
            if self.verbose:
                agents = self.expected_agents if self.expected_agents is not None else sorted(self.episode_successes.keys())
                print("\n" + "=" * 70)
                print(
                    f"TARGET REACHED: ultimi {self.success_window} episodi = 100% success "
                    f"per agenti {agents}. Training interrotto."
                )
                print("=" * 70 + "\n")
            return False

        # Stampa periodica
        if self.n_calls > 0 and self.n_calls % self.check_freq == 0:
            print(f"\n{'='*70}")
            print(f"📊 Progress at {self.n_calls:,} steps")
            print(f"{'='*70}")

            global_trim_eff = []
            global_trim_error = []
            global_vmg = []
            global_speed = []
            global_success_rates = []

            for agent in sorted(self.episode_successes):
                n_recent = len(self.episode_successes[agent])
                if n_recent > 0:
                    n_successes = sum(self.episode_successes[agent])
                    success_rate = (n_successes / n_recent) * 100
                    avg_dist = np.mean(self.episode_distances[agent])
                    print(f"{agent} - Last {n_recent} episodes:")
                    print(f"   ✓ Successes: {n_successes}/{n_recent} ({success_rate:.1f}%)")
                    print(f"   Avg final distance: {avg_dist:.1f} m")
                    global_success_rates.append(success_rate)

                    self.logger.record(f"success/{agent}_rate", success_rate)
                    self.logger.record(f"distance/{agent}_avg_final_m", float(avg_dist))
                else:
                    print(f"{agent} - No episodes completed yet")

                if self.trim_eff_window[agent]:
                    eff_mean = float(np.mean(self.trim_eff_window[agent]))
                    global_trim_eff.append(eff_mean)
                    self.logger.record(f"trim/{agent}_efficiency_mean", eff_mean)
                if self.trim_error_window[agent]:
                    err_mean = float(np.mean(self.trim_error_window[agent]))
                    global_trim_error.append(err_mean)
                    self.logger.record(f"trim/{agent}_error_mean", err_mean)
                if self.vmg_window[agent]:
                    vmg_mean = float(np.mean(self.vmg_window[agent]))
                    global_vmg.append(vmg_mean)
                    self.logger.record(f"vmg/{agent}_mean_kts", vmg_mean)
                if self.speed_window[agent]:
                    speed_mean = float(np.mean(self.speed_window[agent]))
                    global_speed.append(speed_mean)
                    self.logger.record(f"speed/{agent}_mean_kts", speed_mean)

            if global_trim_eff:
                self.logger.record("trim/global_efficiency_mean", float(np.mean(global_trim_eff)))
            if global_trim_error:
                self.logger.record("trim/global_error_mean", float(np.mean(global_trim_error)))
            if global_vmg:
                self.logger.record("vmg/global_mean_kts", float(np.mean(global_vmg)))
            if global_speed:
                self.logger.record("speed/global_mean_kts", float(np.mean(global_speed)))
            if global_success_rates:
                self.logger.record("success/global_rate", float(np.mean(global_success_rates)))

            print(f"{'='*70}\n")

        return True