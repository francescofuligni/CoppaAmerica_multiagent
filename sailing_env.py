import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional

from wind_model import WindField

class ImprovedSailingEnv(ParallelEnv):
    """
    Ambiente di navigazione a vela migliorato, compatibile con PettingZoo Parallel API.
    """
    metadata = {'render_modes': ['rgb_array'], "name": "sailing_v0"}

    def __init__(self, field_size=400, render_mode=None):
        super().__init__()
        
        self.field_size = field_size
        self.max_speed = 15.0
        self.max_wind = 25.0
        self.target_radius = 50.0
        self.dt = 1.0
        self.max_steps = 250
        self.render_mode = render_mode

        # Configurazione PettingZoo
        self.possible_agents = ["boat_0"]
        self.agents = self.possible_agents[:]

        self.observation_spaces = {agent: spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        ) for agent in self.possible_agents}
        
        self.action_spaces = {agent: spaces.Discrete(4) 
                              for agent in self.possible_agents}
        
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

        for agent in self.possible_agents:
            self.state[agent] = {
                'x': self.np_random.uniform(40, 120),
                'y': self.np_random.uniform(40, 120),
                'speed': 0.0,
                'heading': self.np_random.uniform(0, 2*np.pi)
            }
            
            self.target[agent] = np.array([
                self.np_random.uniform(self.field_size - 120, self.field_size - 40),
                self.np_random.uniform(self.field_size - 120, self.field_size - 40)
            ])
            
            self.trajectory[agent] = [np.array([self.state[agent]['x'], self.state[agent]['y']])]
            
            pos = np.array([self.state[agent]['x'], self.state[agent]['y']])
            self.previous_distance[agent] = np.linalg.norm(pos - self.target[agent])
            self.best_distance[agent] = self.previous_distance[agent]
        
        # Inizializziamo il vento con direzione orientata verso il target del primo agente
        ref_agent = self.possible_agents[0]
        target_angle = np.arctan2(
            self.target[ref_agent][1] - self.state[ref_agent]['y'],
            self.target[ref_agent][0] - self.state[ref_agent]['x']
        )
        base_dir = target_angle + np.pi / 2 + self.np_random.uniform(-np.pi / 6, np.pi / 6)
        self.wind_field.reset(self.np_random, base_direction=base_dir)

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        return observations, infos

    def _get_obs(self, agent):
        pos = np.array([self.state[agent]['x'], self.state[agent]['y']])
        dist_to_target = np.linalg.norm(pos - self.target[agent])
        angle_to_target = np.arctan2(self.target[agent][1] - pos[1], self.target[agent][0] - pos[0])
        angle_to_target = self._normalize_angle(angle_to_target)
        
        local_wind_dir, local_wind_speed = self.wind_field.get_local_wind(
            self.state[agent]['x'], self.state[agent]['y']
        )
        obs = np.array([
            self.state[agent]['x'] / self.field_size,
            self.state[agent]['y'] / self.field_size,
            self.state[agent]['speed'] / self.max_speed,
            self.state[agent]['heading'] / (2 * np.pi),
            local_wind_dir / (2 * np.pi),
            local_wind_speed / self.max_wind,
            dist_to_target / (self.field_size * np.sqrt(2)),
            angle_to_target / (2 * np.pi)
        ], dtype=np.float32)
        
        return obs

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        self.step_count += 1
        self.wind_field.step()  # aggiorna il campo di vento ad ogni step

        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        for agent, action in actions.items():
            pos = np.array([self.state[agent]['x'], self.state[agent]['y']])
            prev_dist = np.linalg.norm(pos - self.target[agent])
            
            if action == 0: self.state[agent]['heading'] -= np.radians(15)
            elif action == 2: self.state[agent]['heading'] += np.radians(15)
            elif action == 3: self.state[agent]['heading'] += np.pi
                
            self.state[agent]['heading'] = self._normalize_angle(self.state[agent]['heading'])

            local_wind_dir, local_wind_speed = self.wind_field.get_local_wind(
                self.state[agent]['x'], self.state[agent]['y']
            )
            apparent_wind_angle = local_wind_dir - self.state[agent]['heading']
            self.state[agent]['speed'] = self._get_polar_speed(apparent_wind_angle, local_wind_speed)
            
            displacement = self.state[agent]['speed'] * 0.514 * self.dt
            self.state[agent]['x'] += displacement * np.cos(self.state[agent]['heading'])
            self.state[agent]['y'] += displacement * np.sin(self.state[agent]['heading'])
            
            self.trajectory[agent].append(np.array([self.state[agent]['x'], self.state[agent]['y']]))
            
            pos = np.array([self.state[agent]['x'], self.state[agent]['y']])
            dist_to_target = np.linalg.norm(pos - self.target[agent])
            
            reward = 0.0
            terminated = False
            truncated = False
            
            distance_delta = prev_dist - dist_to_target
            reward += distance_delta * 2.0
            
            if dist_to_target < 200: reward += 10.0
            if dist_to_target < 150: reward += 15.0
            if dist_to_target < 100: reward += 25.0
            if dist_to_target < 75:  reward += 40.0
            if dist_to_target < 60:  reward += 60.0
                
            if dist_to_target > 100: reward -= 0.2
            elif dist_to_target > 60: reward -= 0.1
            else: reward -= 0.05
                
            if distance_delta > 0: reward += self.state[agent]['speed'] * 0.15
                
            angle_to_target = np.arctan2(self.target[agent][1] - pos[1], self.target[agent][0] - pos[0])
            heading_error = abs(self._normalize_angle(angle_to_target - self.state[agent]['heading']))
            if heading_error > np.pi: heading_error = 2 * np.pi - heading_error
            heading_alignment = 1.0 - (heading_error / np.pi)
            
            proximity_factor = 1.0
            if dist_to_target < 100: proximity_factor = 2.0
            if dist_to_target < 60:  proximity_factor = 3.0
            reward += heading_alignment * 1.5 * proximity_factor
            
            if dist_to_target < self.best_distance[agent]:
                improvement = self.best_distance[agent] - dist_to_target
                reward += improvement * 1.0
                self.best_distance[agent] = dist_to_target
                
            if dist_to_target < self.target_radius:
                efficiency_bonus = max(0, 250 - self.step_count) * 2
                reward += 2000.0 + efficiency_bonus
                terminated = True
                
            if (self.state[agent]['x'] < 0 or self.state[agent]['x'] > self.field_size or
                self.state[agent]['y'] < 0 or self.state[agent]['y'] > self.field_size):
                reward -= 100.0
                terminated = True
                
            if dist_to_target > prev_dist + 25: reward -= 5.0
                
            if self.step_count >= self.max_steps:
                truncated = True
                if self.best_distance[agent] < 60: reward += 200.0
                elif self.best_distance[agent] < 100: reward += 100.0
                elif self.best_distance[agent] < 150: reward += 50.0
                    
            observations[agent] = self._get_obs(agent)
            rewards[agent] = reward
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = {
                'distance_to_target': dist_to_target,
                'speed': self.state[agent]['speed'],
                'steps': self.step_count,
                'best_distance': self.best_distance[agent]
            }

        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]
        
        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()
            
    def _render_frame(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.field_size)
        ax.set_ylim(0, self.field_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # --- Frecce vento (griglia 8x8) ---
        xs, ys, us, vs = self.wind_field.get_grid_arrows(n_arrows=8)
        speeds = np.sqrt(us**2 + vs**2)
        ax.quiver(
            xs, ys, us, vs,
            speeds,
            cmap='Blues', alpha=0.55,
            scale=220, width=0.003,
            headwidth=4, headlength=5,
        )

        for agent in self.possible_agents:
            if agent in self.target:
                target_circle = plt.Circle(self.target[agent], self.target_radius, 
                                          color='red', alpha=0.3, label=f'Target {agent}')
                ax.add_patch(target_circle)
        
        colors = ['blue', 'orange', 'purple', 'brown']
        idx = 0
        for agent in self.possible_agents:
            if agent in self.trajectory and len(self.trajectory[agent]) > 1:
                traj = np.array(self.trajectory[agent])
                ax.plot(traj[:, 0], traj[:, 1], '-', color=colors[idx%len(colors)], alpha=0.5, linewidth=2)
                
            if agent in self.state:
                boat_size = 15
                boat_points = np.array([[boat_size, 0], [-boat_size/2, boat_size/2], [-boat_size/2, -boat_size/2]])
                rotation_matrix = np.array([
                    [np.cos(self.state[agent]['heading']), -np.sin(self.state[agent]['heading'])],
                    [np.sin(self.state[agent]['heading']), np.cos(self.state[agent]['heading'])]
                ])
                boat_points = boat_points @ rotation_matrix.T
                boat_points += np.array([self.state[agent]['x'], self.state[agent]['y']])
                
                boat = patches.Polygon(boat_points, closed=True, color='green', 
                                      edgecolor='darkgreen', linewidth=2)
                ax.add_patch(boat)
            idx += 1
        
        ref_agent = self.possible_agents[0]
        if ref_agent in self.state:
            dist = np.linalg.norm(np.array([self.state[ref_agent]['x'], self.state[ref_agent]['y']]) - self.target[ref_agent])

            # Vento locale nella posizione della barca
            local_wd, local_ws = self.wind_field.get_local_wind(
                self.state[ref_agent]['x'], self.state[ref_agent]['y']
            )
            wind_deg = np.degrees(local_wd) % 360

            ax.set_title(
                f"Step: {self.step_count} | Boat speed: {self.state[ref_agent]['speed']:.1f} kts | "
                f"Distance: {dist:.0f}m (Best: {self.best_distance[ref_agent]:.0f}m)",
                fontsize=10, weight='bold'
            )

            # --- Box info vento (angolo in alto a sinistra) ---
            wind_text = (
                f"Wind (base)\n"
                f"Dir: {np.degrees(self.wind_field.base_direction) % 360:.0f}°\n"
                f"Speed: {self.wind_field.base_speed:.1f} kts\n"
                f"\nWind (local @ boat)\n"
                f"Dir: {wind_deg:.0f}°\n"
                f"Speed: {local_ws:.1f} kts"
            )
            ax.text(
                0.02, 0.98, wind_text,
                transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85, edgecolor='gray'),
                family='monospace'
            )

            # --- Rosa dei venti compatta (angolo in alto a destra) ---
            inset_ax = fig.add_axes([0.78, 0.78, 0.16, 0.16], polar=True)
            inset_ax.set_theta_zero_location('N')
            inset_ax.set_theta_direction(-1)
            inset_ax.set_rticks([])
            inset_ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
            inset_ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=6)
            inset_ax.set_title('Wind', fontsize=7, pad=2)
            # Freccia vento base (blu)
            inset_ax.annotate(
                '', xy=(self.wind_field.base_direction, 0.8), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.8)
            )
            # Freccia vento locale (arancione tratteggiata)
            inset_ax.annotate(
                '', xy=(local_wd, 0.65), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.4, linestyle='dashed')
            )
        
        fig.canvas.draw()
        try:
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        except AttributeError:
             image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
             image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
             
        plt.close(fig)
        return image
        
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
