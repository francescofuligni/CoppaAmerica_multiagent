import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional

from wind_model import WindField

class ImprovedSailingEnv(ParallelEnv):
    """
    Ambiente di navigazione a vela migliorato con due barche, compatibile con PettingZoo Parallel API.
    """
    metadata = {'render_modes': ['rgb_array'], "name": "sailing_v0"}

    def __init__(self, field_size=400, render_mode=None):
        super().__init__()
        
        self.field_size = field_size
        self.max_speed = 15.0
        self.max_wind = 30.0
        self.target_radius = 50.0
        self.dt = 1.0
        self.max_steps = 400
        self.render_mode = render_mode

        # Ora due barche
        self.possible_agents = ["red_boat", "blue_boat"]
        self.agents = self.possible_agents[:]

        # Timone continuo
        self.max_turn_per_step = np.radians(25)
        self.min_turn_factor = 0.12
        self.speed_inertia = 0.85

        # Obs: (x, y, speed, sin_h, cos_h, sin_aw, cos_aw, wind_speed, dist, sin_rb, cos_rb, rudder)
        self.observation_spaces = {agent: spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        ) for agent in self.possible_agents}

        self.action_spaces = {agent: spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        ) for agent in self.possible_agents}
        
        self.state = {}
        self.target = {}
        self.wind_field = WindField(field_size=field_size)
        self.step_count = 0
        self.trajectory = {}
        self.previous_distance = {}
        self.best_distance = {}
        self.fig = None
        self.ax = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_polar_speed(self, apparent_wind_angle, wind_speed):
        angle_deg = np.abs(np.degrees(apparent_wind_angle) % 360)
        if angle_deg > 180: angle_deg = 360 - angle_deg
            
        if angle_deg < 40: speed_ratio = 0.0
        elif angle_deg < 50: speed_ratio = 0.2 + (angle_deg - 40) * 0.02
        elif angle_deg < 90: speed_ratio = 0.4 + (angle_deg - 50) * 0.0075
        elif angle_deg < 120: speed_ratio = 0.7
        elif angle_deg < 150: speed_ratio = 0.7 - (angle_deg - 120) * 0.003
        else: speed_ratio = 0.6 - (angle_deg - 150) * 0.005
            
        return min(speed_ratio * wind_speed, self.max_speed)

    def _normalize_angle(self, angle):
        return angle % (2 * np.pi)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        elif not hasattr(self, 'np_random'):
            self.np_random, _ = gym.utils.seeding.np_random()
            
        self.agents = self.possible_agents[:]
        self.step_count = 0

        # --- crea linee di partenza e arrivo ---
        self.start_line_y = 50.0
        self.start_line_x = (40.0, 100.0)
        
        self.finish_line_y = 350.0
        self.finish_line_x = (300.0, 360.0)
        
        shared_target = np.array([330.0, self.finish_line_y])
        
        initial_heading = np.arctan2(self.finish_line_y - self.start_line_y, 330.0 - 70.0)
        
        for i, agent in enumerate(self.possible_agents):
            self.state[agent] = {
                'x': 55.0 if i == 0 else 85.0,
                'y': self.start_line_y,
                'speed': 0.0,
                'heading': initial_heading,
                'rudder_angle': 0.0,
            }
            
            # usa lo stesso target per tutte le barche
            self.target[agent] = shared_target.copy()
            
            self.trajectory[agent] = [np.array([self.state[agent]['x'], self.state[agent]['y']])]
            pos = np.array([self.state[agent]['x'], self.state[agent]['y']])
            self.previous_distance[agent] = np.linalg.norm(pos - self.target[agent])
            self.best_distance[agent] = self.previous_distance[agent]
        
        # Vento
        if options and 'wind_direction' in options:
            base_dir = float(options['wind_direction'])
        else:
            base_dir = float(self.np_random.uniform(0, 2 * np.pi))
        self.wind_field.reset(self.np_random, base_direction=base_dir)

        # Init heading random rimosso per mantenere quello perpendicolare fisso alla linea di partenza

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        return observations, infos

    def _get_obs(self, agent):
        pos = np.array([self.state[agent]['x'], self.state[agent]['y']])
        dist_to_target = np.linalg.norm(pos - self.target[agent])
        bearing_to_target = np.arctan2(
            self.target[agent][1] - pos[1], self.target[agent][0] - pos[0]
        )
        
        local_wind_dir, local_wind_speed = self.wind_field.get_local_wind(
            self.state[agent]['x'], self.state[agent]['y']
        )
        
        heading = self.state[agent]['heading']
        rel_bearing = bearing_to_target - heading
        wind_from_dir = local_wind_dir + np.pi
        apparent_wind = wind_from_dir - heading
        
        obs = np.array([
            self.state[agent]['x'] / self.field_size,
            self.state[agent]['y'] / self.field_size,
            self.state[agent]['speed'] / self.max_speed,
            np.sin(heading), np.cos(heading),
            np.sin(apparent_wind), np.cos(apparent_wind),
            local_wind_speed / self.max_wind,
            dist_to_target / (self.field_size * np.sqrt(2)),
            np.sin(rel_bearing), np.cos(rel_bearing),
            float(self.state[agent]['rudder_angle']),
        ], dtype=np.float32)
        
        return np.clip(obs, -1.0, 1.0)

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        self.step_count += 1
        self.wind_field.step()

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent, action in actions.items():
            # --- salta agenti già terminati ---
            if agent not in self.agents:
                continue

            pos = np.array([self.state[agent]['x'], self.state[agent]['y']])
            prev_dist = np.linalg.norm(pos - self.target[agent])

            rudder_input = float(np.clip(action[0] if hasattr(action, '__len__') else action, -1.0, 1.0))
            self.state[agent]['rudder_angle'] = rudder_input

            # calcola heading e velocità
            speed_factor = self.state[agent]['speed'] / self.max_speed
            effective_factor = self.min_turn_factor + (1.0 - self.min_turn_factor) * speed_factor
            turn_rate = rudder_input * effective_factor * self.max_turn_per_step
            self.state[agent]['heading'] = self._normalize_angle(
                self.state[agent]['heading'] + turn_rate * self.dt
            )

            brake_factor = min(1.0, abs(rudder_input) * 0.5)
            self.state[agent]['speed'] *= (1.0 - brake_factor * 0.2)

            local_wind_dir, local_wind_speed = self.wind_field.get_local_wind(
                self.state[agent]['x'], self.state[agent]['y']
            )
            wind_from_dir = local_wind_dir + np.pi
            apparent_wind_angle = wind_from_dir - self.state[agent]['heading']
            target_speed = self._get_polar_speed(apparent_wind_angle, local_wind_speed)
            self.state[agent]['speed'] = self.state[agent]['speed'] * self.speed_inertia + target_speed * (1.0 - self.speed_inertia)

            # sposta la barca
            displacement = self.state[agent]['speed'] * 0.514 * self.dt
            self.state[agent]['x'] += displacement * np.cos(self.state[agent]['heading'])
            self.state[agent]['y'] += displacement * np.sin(self.state[agent]['heading'])

            self.trajectory[agent].append(np.array([self.state[agent]['x'], self.state[agent]['y']]))

            # distanza dal target
            pos = np.array([self.state[agent]['x'], self.state[agent]['y']])
            dist_to_target = np.linalg.norm(pos - self.target[agent])

            reward = 0.0
            terminated = False
            truncated = False

            distance_delta = prev_dist - dist_to_target
            reward += distance_delta * 5.0
            reward -= 0.3
            reward -= abs(rudder_input) * 0.1
            if distance_delta > 0:
                reward += self.state[agent]['speed'] * 0.1

            if dist_to_target < self.best_distance[agent]:
                self.best_distance[agent] = dist_to_target

            # --- TERMINAZIONE INDIVIDUALE ---
            crossed_finish = (self.state[agent]['y'] >= self.finish_line_y and
                              self.finish_line_x[0] <= self.state[agent]['x'] <= self.finish_line_x[1])
            
            if crossed_finish:
                efficiency = max(0, self.max_steps - self.step_count) / self.max_steps
                reward += 1000.0 + efficiency * 500.0
                terminated = True
                # salva step individuale
                self.state[agent]['steps_to_target'] = self.step_count

            elif (self.state[agent]['x'] < 0 or self.state[agent]['x'] > self.field_size or
                self.state[agent]['y'] < 0 or self.state[agent]['y'] > self.field_size or
                self.state[agent]['y'] >= self.finish_line_y + 10.0): # Uscito dal campo o mancato il gate
                reward -= 150.0
                terminated = True

            if self.step_count >= self.max_steps:
                truncated = True
                progress = 1.0 - (self.best_distance[agent] / max(self.previous_distance[agent], 1.0))
                if progress > 0:
                    reward += progress * 200.0

            # aggiorna dizionari
            observations[agent] = self._get_obs(agent)
            rewards[agent] = reward
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = {
                'distance_to_target': dist_to_target,
                'speed': self.state[agent]['speed'],
                'steps': self.step_count,
                'steps_to_target': self.state[agent].get('steps_to_target', None),
                'best_distance': self.best_distance[agent]
            }

        # rimuovi agenti terminati dalla lista attiva
        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]

        return observations, rewards, terminations, truncations, infos


    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def _render_frame(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8,8))

        self.fig.clf() # Fully clear the figure to avoid frame overlapping
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, self.field_size)
        self.ax.set_ylim(0, self.field_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#a0d8ef')

        # --- Frecce vento ---
        xs, ys, us, vs = self.wind_field.get_grid_arrows(n_arrows=8)
        speeds = np.sqrt(us**2 + vs**2)
        self.ax.quiver(xs, ys, us, vs, speeds, cmap='Blues', alpha=0.55,
                    scale=220, width=0.003, headwidth=4, headlength=5)

        colors_map = {'red_boat': 'red', 'blue_boat': 'blue'}

        # --- Start / Finish Line ---
        if hasattr(self, 'start_line_y'):
            # Linea di partenza
            self.ax.plot([self.start_line_x[0], self.start_line_x[1]], 
                         [self.start_line_y, self.start_line_y], 
                         'k--', linewidth=2, alpha=0.6, label='Start')
                         
            # Linea di arrivo (verde trasparente)
            self.ax.plot([self.finish_line_x[0], self.finish_line_x[1]], 
                         [self.finish_line_y, self.finish_line_y], 
                         'g-', linewidth=4, alpha=0.4, label='Finish')

        # --- Traiettorie e barche ---
        for idx, agent in enumerate(self.possible_agents):
            agent_color = colors_map.get(agent, 'black')
            
            # Traiettoria
            if agent in self.trajectory and len(self.trajectory[agent]) > 1:
                traj = np.array(self.trajectory[agent])
                self.ax.plot(traj[:,0], traj[:,1], '-', color=agent_color,
                            alpha=0.5, linewidth=2)

            # Barca
            if agent in self.state:
                bx, by = self.state[agent]['x'], self.state[agent]['y']
                hdg = self.state[agent]['heading']
                boat_size = 15
                points = np.array([[boat_size,0], [-boat_size/2,boat_size/2], [-boat_size/2,-boat_size/2]])
                rot = np.array([[np.cos(hdg), -np.sin(hdg)],
                                [np.sin(hdg),  np.cos(hdg)]])
                points = points @ rot.T + np.array([bx, by])
                edge_color = 'darkred' if agent_color == 'red' else 'darkblue'
                boat = patches.Polygon(points, closed=True, facecolor=agent_color, edgecolor=edge_color, linewidth=2)
                self.ax.add_patch(boat)

        # --- Info dinamiche: distanza e step per ogni barca ---
        info_lines = []
        for agent in self.possible_agents:
            if agent in self.state:
                bx, by = self.state[agent]['x'], self.state[agent]['y']
                dist = np.linalg.norm(np.array([bx, by]) - self.target[agent])
                steps = self.state[agent].get('steps_to_target', self.step_count)
                info_lines.append(f"{agent}: distance {dist:.1f} m, {steps} steps")

        # --- Determina il vincitore se almeno una barca ha raggiunto il target ---
        winner_text = ""
        finished_agents = {a: self.state[a]['steps_to_target'] for a in self.possible_agents if 'steps_to_target' in self.state[a]}
        if finished_agents:
            winner_agent = min(finished_agents, key=finished_agents.get)
            winner_steps = finished_agents[winner_agent]
            winner_text = f"Vincitore: {winner_agent} ({winner_steps} steps)\n"

        # Combina vincitore + info linee
        full_title = winner_text + " | ".join(info_lines)
        self.ax.set_title(full_title, fontsize=10)

        # Converti in immagine RGB
        self.fig.canvas.draw()
        img = np.array(self.fig.canvas.buffer_rgba(), copy=True)
        img = img[:, :, :3]
        return img

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)