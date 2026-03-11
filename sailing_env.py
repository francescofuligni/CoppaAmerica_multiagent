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

    def __init__(self, field_size=2500, render_mode=None):
        super().__init__()
        
        self.field_size = field_size
        self.max_speed = 50.0  # Increased for foiling speeds
        self.max_wind = 30.0
        self.target_radius = 50.0  # Kept small relative to field to require precision
        self.dt = 1.0
        self.max_steps = 2500
        self.render_mode = render_mode

        # Ora due barche
        self.possible_agents = ["red_boat", "blue_boat"]
        self.agents = self.possible_agents[:]

        # Timone continuo
        self.max_turn_per_step = np.radians(25)
        self.min_turn_factor = 0.12
        # Inerzia velocità (momentum). Altissima sui foil (scivola nell'aria), bassa in acqua
        self.displacement_inertia = 0.85
        self.foiling_inertia = 0.98

        # Foiling characteristics
        self.foiling_takeoff_speed = 18.0  # Takeoff threshold as requested (18 kts)
        self.foiling_drop_speed = 15.0     # Hysteresis for dropping off the foil (15 kts)

        # --- Costanti del Campo di Regata (Windward-Leeward) ---
        self.course_center_x = self.field_size / 2.0
        self.boundaries = {'x_min': 500.0, 'x_max': 2000.0}
        self.top_gate_y = 2300.0
        self.bottom_gate_y = 200.0
        self.gate_width = 300.0

        # Obs: (x, y, speed, sin_h, cos_h, sin_aw, cos_aw, wind_speed, dist, sin_rb, cos_rb, rudder, is_foiling, active_foil, is_upwind_leg)
        self.observation_spaces = {agent: spaces.Box(
            low=-1.0, high=1.0, shape=(15,), dtype=np.float32
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

    def _get_polar_speed(self, apparent_wind_angle, wind_speed, is_foiling):
        # NOTA: apparent_wind_angle e' in realtà il True Wind Angle (TWA)!
        diff = apparent_wind_angle % (2 * np.pi)
        if diff > np.pi:
            diff = 2 * np.pi - diff
        angle_deg = np.degrees(diff)
            
        if is_foiling:
            # Foiling (AC75-like): Impossibile stringere il vento puro, si vola solo dai 45-50° in sù (bolina larga/traverso)
            if angle_deg < 45: speed_ratio = 0.0
            elif angle_deg < 55: speed_ratio = 1.0 + (angle_deg - 45) * 0.15
            elif angle_deg < 100: speed_ratio = 2.5 + (angle_deg - 55) * 0.024
            elif angle_deg < 140: speed_ratio = 3.6 + (angle_deg - 100) * 0.015
            elif angle_deg < 170: speed_ratio = 4.2 - (angle_deg - 140) * 0.03  # Veloci in poppa fino a 170° (cala dolcemente a 3.3x)
            else: speed_ratio = 3.3 - (angle_deg - 170) * 0.25 # Stalla bruscamente solo oltre i 170° per l'ombra del vento
        else:
            # Displacement: rotta in acqua, l'andatura di poppa funziona ma è molto più lenta del foiling
            if angle_deg < 35: speed_ratio = 0.0
            elif angle_deg < 50: speed_ratio = 0.3 + (angle_deg - 35) * 0.03
            elif angle_deg < 110: speed_ratio = 0.75 + (angle_deg - 50) * 0.015
            elif angle_deg < 140: speed_ratio = 1.65 - (angle_deg - 110) * 0.01
            else: speed_ratio = 1.35 - (angle_deg - 140) * 0.005 # A 180 gradi non stalla, viaggia tranquilla in acqua
            
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

        # Inizializziamo il vento fisicamente da Nord a Sud (asse Y dall'alto verso il basso)
        if options and 'wind_direction' in options:
            base_dir = float(options['wind_direction'])
        else:
            base_dir = 1.5 * np.pi
        self.wind_field.reset(self.np_random, base_direction=base_dir)

        # Ora posizioniamo barca e primo target (Top Gate)
        for i, agent in enumerate(self.possible_agents):
            # Spawn barca tra Y=0 e Y=250 all'interno del Boundary X (distanziate verticalmente leggermente)
            start_x = self.np_random.uniform(self.boundaries['x_min'] + 50, self.boundaries['x_max'] - 50)
            start_y = self.np_random.uniform(20.0, 100.0) + i * 20.0 
            
            self.state[agent] = {
                'x': start_x,
                'y': start_y,
                'speed': 0.0,
                'heading': 0.0, # Verrà sovrascritto
                'rudder_angle': 0.0,
                'is_foiling': False,
                'active_foil': 1.0,
                'dropped_foil_penalty_applied': False,
                'current_leg': 1 # 1: Bolina, 2: Poppa
            }
            
            # Bersaglio Iniziale: centro del Top Gate
            self.target[agent] = np.array([self.course_center_x, self.top_gate_y])
            
            self.trajectory[agent] = [np.array([self.state[agent]['x'], self.state[agent]['y']])]
            pos = np.array([self.state[agent]['x'], self.state[agent]['y']])
            self.previous_distance[agent] = np.linalg.norm(pos - self.target[agent])
            self.best_distance[agent] = self.previous_distance[agent]
        # Inizializza heading casuale per bolina
        for agent in self.possible_agents:
            tack_sign = float(self.np_random.choice([-1, 1]))
            start_heading = base_dir + np.pi + tack_sign * np.radians(50.0)
            start_heading += self.np_random.uniform(-np.radians(10), np.radians(10))
            self.state[agent]['heading'] = self._normalize_angle(start_heading)

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
        # Angolo apparente del vento rispetto alla prua, simmetrico da -pi a +pi
        apparent_wind = (local_wind_dir + np.pi) - heading
        apparent_wind = self._normalize_angle(apparent_wind)
        if apparent_wind > np.pi:
            apparent_wind -= 2 * np.pi
        
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
            1.0 if self.state[agent]['is_foiling'] else 0.0,     # [0, 1] boolean foiling state
            self.state[agent]['active_foil'],                    # [-1, 1] Port vs Starboard foil
            1.0 if self.state[agent]['current_leg'] == 1 else -1.0 # [-1, 1] Leg information
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
            
            prev_rudder = self.state[agent].get('rudder_angle', 0.0)

            rudder_input = float(np.clip(action[0] if hasattr(action, '__len__') else action, -1.0, 1.0))
            self.state[agent]['rudder_angle'] = rudder_input

            # calcola heading e velocità
            speed_factor = self.state[agent]['speed'] / self.max_speed
            effective_factor = self.min_turn_factor + (1.0 - self.min_turn_factor) * speed_factor
            turn_rate = rudder_input * effective_factor * self.max_turn_per_step
            self.state[agent]['heading'] = self._normalize_angle(
                self.state[agent]['heading'] + turn_rate * self.dt
            )

            # 3. Frenata virata brusca: ridotta in foiling per permettere virate/strambate più vere e lunghe
            if self.state[agent]['is_foiling']:
                brake_factor = (abs(rudder_input) ** 1.5) * 0.05
                self.state[agent]['speed'] *= (1.0 - brake_factor)
            else:
                brake_factor = (abs(rudder_input) ** 1.5) * 0.35
                self.state[agent]['speed'] *= (1.0 - brake_factor)

            local_wind_dir, local_wind_speed = self.wind_field.get_local_wind(
                self.state[agent]['x'], self.state[agent]['y']
            )
            apparent_wind_angle = (local_wind_dir + np.pi) - self.state[agent]['heading']
            target_speed = self._get_polar_speed(apparent_wind_angle, local_wind_speed, self.state[agent]['is_foiling'])
            
            current_inertia = self.foiling_inertia if self.state[agent]['is_foiling'] else self.displacement_inertia
            self.state[agent]['speed'] = self.state[agent]['speed'] * current_inertia + target_speed * (1.0 - current_inertia)
            
            # 5. Foil mechanics
            was_foiling = self.state[agent]['is_foiling']
            
            aw_norm = self._normalize_angle(apparent_wind_angle)
            if aw_norm > np.pi:
                aw_norm -= 2 * np.pi
            self.state[agent]['active_foil'] = 1.0 if aw_norm > 0 else -1.0

            if self.state[agent]['speed'] >= self.foiling_takeoff_speed:
                self.state[agent]['is_foiling'] = True
            elif self.state[agent]['speed'] < self.foiling_drop_speed:
                self.state[agent]['is_foiling'] = False

            dropped_foil = was_foiling and not self.state[agent]['is_foiling']
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
            # 1. Progresso verso target e VMG (Velocity Made Good)
            distance_delta = prev_dist - dist_to_target
            
            if self.state[agent]['is_foiling']:
                if distance_delta > 0:
                    reward += distance_delta * ((self.state[agent]['speed'] / 15.0) ** 2)
                else:
                    reward += distance_delta * 0.1
            else:
                reward += distance_delta * 0.1

            # 2. Costo per step (urgenza)
            reward -= 0.5

            # 3. Penalità timone logaritmica/esponenziale (riduce strambate secche)
            reward -= (abs(rudder_input) ** 3) * 15.0
            rudder_delta = rudder_input - prev_rudder
            reward -= (abs(rudder_delta) ** 2) * 50.0

            # 3.b Drop off foil penalty (massive cost - as requested by user)
            if dropped_foil:
                reward -= 300.0  # Costo catastrofico per forzare a girare largo anziché cadere

            # 4. Foiling and SPEED INCENTIVES (Primary focus of the agent)
            if self.state[agent]['is_foiling']:
                speed_over_threshold = self.state[agent]['speed'] - self.foiling_drop_speed
                reward += (speed_over_threshold * 0.8)
            else:
                reward -= 1.0
                
            if self.state[agent]['speed'] < self.foiling_drop_speed:
                reward -= 2.0  # Leggera punizione continua, meno letale per consentire le manovre di recupero
                
            # Bonus velocità assoluta quando ci si avvicina (secondary)
            if distance_delta > 0:
                reward += self.state[agent]['speed'] * 0.2

            if dist_to_target < self.best_distance[agent]:
                self.best_distance[agent] = dist_to_target

            # 6. Gate passing & Boundary logic
            bx = self.state[agent]['x']
            by = self.state[agent]['y']
            
            # Boundary penalty continua: se esce dal corridoio virtuale, paga pesantemente a ogni step
            if bx < self.boundaries['x_min'] or bx > self.boundaries['x_max']:
                reward -= 50.0  # Costringe l'agente a virare PRIMA del limite
                
            # Fuori mappa totale (Game Over se sfonda perfino i bordi della telecamera estesa)
            if bx < 0 or bx > self.field_size or by < 0 or by > self.field_size:
                reward -= 300.0
                terminated = True
                # salva step individuale
                self.state[agent]['steps_to_target'] = self.step_count

            # Controllo attraversamento Cancello (Gate)
            gate_left = self.course_center_x - self.gate_width / 2.0
            gate_right = self.course_center_x + self.gate_width / 2.0
            
            if self.state[agent]['current_leg'] == 1:
                # Upwind: deve superare la Y del Top Gate, restando in mezzo alle boe X
                if by >= self.top_gate_y and gate_left <= bx <= gate_right:
                    self.state[agent]['current_leg'] = 2
                    reward += 500.0  # Super Bonus per aver girato la boa di bolina
                    self.target[agent] = np.array([self.course_center_x, self.bottom_gate_y])
                    # Reset distanze per il nuovo lato (Poppa)
                    self.previous_distance[agent] = np.linalg.norm(pos - self.target[agent])
                    self.best_distance[agent] = self.previous_distance[agent]
                    
            elif self.state[agent]['current_leg'] == 2:
                # Downwind: deve scendere sotto la Y del Bottom Gate, restando in mezzo alle boe X
                if by <= self.bottom_gate_y and gate_left <= bx <= gate_right:
                    efficiency = max(0, self.max_steps - self.step_count) / self.max_steps
                    reward += 2000.0 + efficiency * 1000.0 # Vittoria finale della regata
                    terminated = True
                    self.state[agent]['steps_to_target'] = self.step_count

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
        
        # --- Disegna Boundaries (Confini) ---
        self.ax.plot([self.boundaries['x_min'], self.boundaries['x_min']], [0, self.field_size], 'r--', linewidth=2, alpha=0.5, label='Boundary')
        self.ax.plot([self.boundaries['x_max'], self.boundaries['x_max']], [0, self.field_size], 'r--', linewidth=2, alpha=0.5)

        # --- Disegna Gates (Cancelli) ---
        gate_left = self.course_center_x - self.gate_width / 2.0
        gate_right = self.course_center_x + self.gate_width / 2.0
        
        # Top Gate (Bolina)
        self.ax.plot([gate_left, gate_right], [self.top_gate_y, self.top_gate_y], 'g--', alpha=0.3)
        self.ax.plot(gate_left, self.top_gate_y, 'go', markersize=8, label='Gate Mark')
        self.ax.plot(gate_right, self.top_gate_y, 'go', markersize=8)
        
        # Bottom Gate (Poppa)
        self.ax.plot([gate_left, gate_right], [self.bottom_gate_y, self.bottom_gate_y], 'g--', alpha=0.3)
        self.ax.plot(gate_left, self.bottom_gate_y, 'bo', markersize=8)
        self.ax.plot(gate_right, self.bottom_gate_y, 'bo', markersize=8)

        info_lines = []
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
                boat_points = np.array([[boat_size,0], [-boat_size/2,boat_size/2], [-boat_size/2,-boat_size/2]])
                rot = np.array([[np.cos(hdg), -np.sin(hdg)],
                                [np.sin(hdg),  np.cos(hdg)]])
                boat_points = boat_points @ rot.T + np.array([bx, by])
                
                # Highlight in foil
                boat_color = 'cyan' if self.state[agent].get('is_foiling', False) else agent_color
                boat_edge = agent_color if self.state[agent].get('is_foiling', False) else 'darkgray'
                boat = patches.Polygon(boat_points, closed=True,
                                      facecolor=boat_color, edgecolor=boat_edge, linewidth=2)
                self.ax.add_patch(boat)
                
                # Active foil text indication
                foil_side = "Port" if self.state[agent].get('active_foil', 1.0) == 1.0 else "Stbd"
                foil_str = f"FOILING ({foil_side})" if self.state[agent].get('is_foiling', False) else f"HULL ({foil_side})"
                self.ax.text(bx, by - 25, foil_str, fontsize=8, color='magenta', fontweight='bold', ha='center')
                
                # Distanza e step infos
                dist = np.linalg.norm(np.array([bx, by]) - self.target[agent])
                steps = self.state[agent].get('steps_to_target', self.step_count)
                info_lines.append(f"{agent}: {dist:.0f}m ({steps}stp)")
                
                rudder_input = self.state[agent].get('rudder_angle', 0.0)
                rudder_deg = int(rudder_input * 25)
                rudder_color = 'red' if abs(rudder_input) > 0.5 else 'black'
                self.ax.text(bx + 18, by + 18,
                        f"{rudder_deg:+d}°",
                        fontsize=7, color=rudder_color, fontweight='bold')
        
        # Info UI
        ref_agent = self.possible_agents[0]
        if ref_agent in self.state:
            dist = np.linalg.norm(np.array([self.state[ref_agent]['x'], self.state[ref_agent]['y']]) - self.target[ref_agent])

            # Vento locale
            local_wd, local_ws = self.wind_field.get_local_wind(
                self.state[ref_agent]['x'], self.state[ref_agent]['y']
            )
            wind_deg = (90 - np.degrees(local_wd)) % 360

            dist_to_gate = abs(self.target[ref_agent][1] - self.state[ref_agent]['y'])
            
            winner_text = ""
            finished_agents = {a: self.state[a]['steps_to_target'] for a in self.possible_agents if 'steps_to_target' in self.state[a]}
            if finished_agents:
                winner_agent = min(finished_agents, key=finished_agents.get)
                winner_steps = finished_agents[winner_agent]
                winner_text = f"WIN: {winner_agent} | "
                
            full_title = winner_text + " | ".join(info_lines)
            
            self.ax.set_title(
                f"{full_title}\n"
                f"Leg: {self.state[ref_agent].get('current_leg', 1)}/2 | "
                f"Speed: {self.state[ref_agent]['speed']:.1f} kts | Dist Y to Gate: {dist_to_gate:.0f}m",
                fontsize=10, weight='bold'
            )

            # --- Box info vento ---
            wind_text = (
                f"Wind (base)\n"
                f"Dir: {(90 - np.degrees(self.wind_field.base_direction)) % 360:.0f}\u00b0\n"
                f"Speed: {self.wind_field.base_speed:.1f} kts\n"
                f"\nWind (local @ boat)\n"
                f"Dir: {wind_deg:.0f}°\n"
                f"Speed: {local_ws:.1f} kts"
            )
            self.ax.text(
                0.02, 0.98, wind_text,
                transform=self.ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85, edgecolor='gray'),
                family='monospace'
            )

            # --- Rosa dei venti ---
            inset_ax = self.fig.add_axes([0.78, 0.78, 0.16, 0.16], polar=True)
            inset_ax.set_theta_zero_location('N')
            inset_ax.set_theta_direction(-1)
            inset_ax.set_rticks([])
            inset_ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
            inset_ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=6)
            inset_ax.set_title('Wind', fontsize=7, pad=2)
            
            compass_base = np.pi / 2 - self.wind_field.base_direction
            compass_local = np.pi / 2 - local_wd
            
            inset_ax.annotate(
                '', xy=(compass_base, 0.8), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.8)
            )
            inset_ax.annotate(
                '', xy=(compass_local, 0.65), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.4, linestyle='dashed')
            )
        
        self.fig.canvas.draw()
        try:
            # Need copy=True so the array doesn't reference the mutable buffer
            image = np.array(self.fig.canvas.buffer_rgba(), copy=True)[:, :, :3]
        except AttributeError:
             image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
             image = np.array(image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,)), copy=True)
             
        return image
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)