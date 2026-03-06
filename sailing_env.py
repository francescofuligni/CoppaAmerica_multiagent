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
    Ambiente di navigazione a vela migliorato, compatibile con PettingZoo Parallel API.
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

        # Configurazione PettingZoo
        self.possible_agents = ["boat_0"]
        self.agents = self.possible_agents[:]

        # Timone continuo: azione in [-1, 1] → da -25° a +25°
        self.max_turn_per_step = np.radians(25)
        # Virata minima garantita anche a velocità zero (evita deadlock no-go zone)
        self.min_turn_factor = 0.12
        # Inerzia velocità: smooth tra vecchia e target (idea del collega)
        self.speed_inertia = 0.85

        # Obs: (x, y, speed, sin_h, cos_h, sin_aw, cos_aw, wind_speed, dist, sin_rb, cos_rb, rudder)
        self.observation_spaces = {agent: spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        ) for agent in self.possible_agents}

        # Timone continuo: float in [-1, 1]
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

        for agent in self.possible_agents:
            self.state[agent] = {
                'x': self.np_random.uniform(40, 120),
                'y': self.np_random.uniform(40, 120),
                'speed': 0.0,
                'heading': self.np_random.uniform(0, 2*np.pi),
                'rudder_angle': 0.0,  # raw input [-1, 1]
            }
            
            self.target[agent] = np.array([
                self.np_random.uniform(self.field_size - 120, self.field_size - 40),
                self.np_random.uniform(self.field_size - 120, self.field_size - 40)
            ])
            
            self.trajectory[agent] = [np.array([self.state[agent]['x'], self.state[agent]['y']])]
            
            pos = np.array([self.state[agent]['x'], self.state[agent]['y']])
            self.previous_distance[agent] = np.linalg.norm(pos - self.target[agent])
            self.best_distance[agent] = self.previous_distance[agent]
        
        # Inizializziamo il vento: direzione completamente casuale [0, 2π]
        # per esporre l'agente a tutte le condizioni di vento in training
        if options and 'wind_direction' in options:
            base_dir = float(options['wind_direction'])
        else:
            base_dir = float(self.np_random.uniform(0, 2 * np.pi))
        self.wind_field.reset(self.np_random, base_direction=base_dir)

        # Inizializza heading su beam reach (±90° dal vento) + piccolo rumore
        # Garantisce velocità non-zero fin dal primo step, evita il deadlock no-go zone
        for agent in self.possible_agents:
            beam_reach = base_dir + np.pi / 2 * float(self.np_random.choice([-1, 1]))
            self.state[agent]['heading'] = self._normalize_angle(
                beam_reach + self.np_random.uniform(-np.radians(20), np.radians(20))
            )

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
        # Angolo relativo: target rispetto alla prua
        rel_bearing = bearing_to_target - heading
        # Angolo apparente del vento rispetto alla prua
        apparent_wind = local_wind_dir - heading
        
        obs = np.array([
            self.state[agent]['x'] / self.field_size,            # [0, 1]
            self.state[agent]['y'] / self.field_size,            # [0, 1]
            self.state[agent]['speed'] / self.max_speed,         # [0, 1]
            np.sin(heading), np.cos(heading),                    # [-1, 1]
            np.sin(apparent_wind), np.cos(apparent_wind),        # [-1, 1]
            local_wind_speed / self.max_wind,                    # [0, 1]
            dist_to_target / (self.field_size * np.sqrt(2)),     # [0, 1]
            np.sin(rel_bearing), np.cos(rel_bearing),            # [-1, 1]
            float(self.state[agent]['rudder_angle']),             # [-1, 1]
        ], dtype=np.float32)
        
        return np.clip(obs, -1.0, 1.0)

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

            # --- Fisica timone continuo ---
            # 1. Azione continua: float in [-1, 1]
            rudder_input = float(np.clip(action[0] if hasattr(action, '__len__') else action, -1.0, 1.0))
            self.state[agent]['rudder_angle'] = rudder_input

            # 2. Velocità di virata: proporzionale alla velocità + minima garantita
            speed_factor = self.state[agent]['speed'] / self.max_speed
            effective_factor = self.min_turn_factor + (1.0 - self.min_turn_factor) * speed_factor
            turn_rate = rudder_input * effective_factor * self.max_turn_per_step
            self.state[agent]['heading'] = self._normalize_angle(
                self.state[agent]['heading'] + turn_rate * self.dt
            )

            # 3. Frenata virata brusca (dal collega): virate aggressive riducono la velocità
            brake_factor = min(1.0, abs(rudder_input) * 0.5)
            self.state[agent]['speed'] *= (1.0 - brake_factor * 0.2)

            # 4. Velocità target dal vento con inerzia (dal collega)
            local_wind_dir, local_wind_speed = self.wind_field.get_local_wind(
                self.state[agent]['x'], self.state[agent]['y']
            )
            apparent_wind_angle = local_wind_dir - self.state[agent]['heading']
            target_speed = self._get_polar_speed(apparent_wind_angle, local_wind_speed)
            self.state[agent]['speed'] = self.state[agent]['speed'] * self.speed_inertia + target_speed * (1.0 - self.speed_inertia)
            
            displacement = self.state[agent]['speed'] * 0.514 * self.dt
            self.state[agent]['x'] += displacement * np.cos(self.state[agent]['heading'])
            self.state[agent]['y'] += displacement * np.sin(self.state[agent]['heading'])
            
            self.trajectory[agent].append(np.array([self.state[agent]['x'], self.state[agent]['y']]))
            
            pos = np.array([self.state[agent]['x'], self.state[agent]['y']])
            dist_to_target = np.linalg.norm(pos - self.target[agent])
            
            reward = 0.0
            terminated = False
            truncated = False
            
            # 1. Progresso verso target (segnale principale)
            distance_delta = prev_dist - dist_to_target
            reward += distance_delta * 5.0

            # 2. Costo per step (urgenza)
            reward -= 0.3

            # 3. Penalità timone (scoraggia zigzag)
            reward -= abs(rudder_input) * 0.1

            # 4. Bonus velocità quando ci si avvicina
            if distance_delta > 0:
                reward += self.state[agent]['speed'] * 0.1

            # 5. Aggiorna record di avvicinamento
            if dist_to_target < self.best_distance[agent]:
                self.best_distance[agent] = dist_to_target

            # 6. Successo: target raggiunto
            if dist_to_target < self.target_radius:
                efficiency = max(0, self.max_steps - self.step_count) / self.max_steps
                reward += 1000.0 + efficiency * 500.0
                terminated = True

            # 7. Fuori campo
            elif (self.state[agent]['x'] < 0 or self.state[agent]['x'] > self.field_size or
                  self.state[agent]['y'] < 0 or self.state[agent]['y'] > self.field_size):
                reward -= 150.0
                terminated = True

            # 8. Tempo scaduto: credito parziale basato sul progresso
            if self.step_count >= self.max_steps:
                truncated = True
                progress = 1.0 - (self.best_distance[agent] / max(self.previous_distance[agent], 1.0))
                if progress > 0:
                    reward += progress * 200.0
                    
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
                bx = self.state[agent]['x']
                by = self.state[agent]['y']
                hdg = self.state[agent]['heading']
                boat_size = 15

                boat_points = np.array([[boat_size, 0], [-boat_size/2, boat_size/2], [-boat_size/2, -boat_size/2]])
                rotation_matrix = np.array([
                    [np.cos(hdg), -np.sin(hdg)],
                    [np.sin(hdg),  np.cos(hdg)]
                ])
                boat_points = boat_points @ rotation_matrix.T
                boat_points += np.array([bx, by])

                boat = patches.Polygon(boat_points, closed=True,
                                      facecolor='green', edgecolor='darkgreen', linewidth=2)
                ax.add_patch(boat)

                # --- Indicatore timone (linea alla poppa) ---
                rudder_input = self.state[agent].get('rudder_angle', 0.0)  # [-1, 1]
                rudder_visual_angle = rudder_input * np.radians(35)  # scala visiva
                stern_x = bx - boat_size * 0.6 * np.cos(hdg)
                stern_y = by - boat_size * 0.6 * np.sin(hdg)
                rudder_dir = hdg + np.pi + rudder_visual_angle
                rudder_len = 11
                rudder_end_x = stern_x + rudder_len * np.cos(rudder_dir)
                rudder_end_y = stern_y + rudder_len * np.sin(rudder_dir)

                rudder_color = 'red' if abs(rudder_input) > 0.6 else 'darkorange'
                ax.plot([stern_x, rudder_end_x], [stern_y, rudder_end_y],
                        color=rudder_color, linewidth=3, solid_capstyle='round')

                rudder_deg = int(rudder_input * 25)  # converti in gradi equivalenti
                ax.text(bx + 18, by + 18,
                        f"{rudder_deg:+d}°",
                        fontsize=7, color=rudder_color, fontweight='bold')
            idx += 1
        
        ref_agent = self.possible_agents[0]
        if ref_agent in self.state:
            dist = np.linalg.norm(np.array([self.state[ref_agent]['x'], self.state[ref_agent]['y']]) - self.target[ref_agent])

            # Vento locale nella posizione della barca
            local_wd, local_ws = self.wind_field.get_local_wind(
                self.state[ref_agent]['x'], self.state[ref_agent]['y']
            )
            wind_deg = (90 - np.degrees(local_wd)) % 360

            ax.set_title(
                f"Step: {self.step_count} | Boat speed: {self.state[ref_agent]['speed']:.1f} kts | "
                f"Distance: {dist:.0f}m (Best: {self.best_distance[ref_agent]:.0f}m)",
                fontsize=10, weight='bold'
            )

            # --- Box info vento (angolo in alto a sinistra) ---
            wind_text = (
                f"Wind (base)\n"
                f"Dir: {(90 - np.degrees(self.wind_field.base_direction)) % 360:.0f}\u00b0\n"
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
            # Converti da convenzione math (0=Est, CCW) a bussola (0=Nord, CW)
            compass_base = np.pi / 2 - self.wind_field.base_direction
            compass_local = np.pi / 2 - local_wd
            # Freccia vento base (blu)
            inset_ax.annotate(
                '', xy=(compass_base, 0.8), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.8)
            )
            # Freccia vento locale (arancione tratteggiata)
            inset_ax.annotate(
                '', xy=(compass_local, 0.65), xytext=(0, 0),
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
