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
from sail_trim import (
    action_to_trim_level,
    normalize_twa_deg,
    optimal_trim_for_twa,
    trim_efficiency,
    trim_level_to_action,
    trim_speed_multiplier,
)

class ImprovedSailingEnv(ParallelEnv):
    """
    Ambiente di navigazione a vela migliorato con due barche, compatibile con PettingZoo Parallel API.
    """
    metadata = {'render_modes': ['rgb_array'], "name": "sailing_v0"}

    def __init__(self, field_size=2500, render_mode=None):
        super().__init__()

        # Geometria mappa (rettangolare)
        self.field_length = 4075.0
        self.field_width = 1482.0
        self.field_diag = float(np.sqrt(self.field_width ** 2 + self.field_length ** 2))
        # Manteniamo field_size per retrocompatibilita' con eventuale codice esterno.
        self.field_size = max(self.field_width, self.field_length)
        self.max_speed = 50.0  # Increased for foiling speeds
        self.max_wind = 30.0
        self.target_radius = 50.0  # Kept small relative to field to require precision
        self.dt = 1.0
        self.max_steps = 1800
        self.render_mode = render_mode

        # Ora due barche
        self.possible_agents = ["red_boat", "blue_boat"]
        self.agents = self.possible_agents[:]

        # Timone continuo
        self.max_turn_per_step = np.radians(25)
        self.min_turn_factor = 0.12
        self.mark_round_margin = 25.0
        self.post_round_offset_x = 90.0
        self.post_round_offset_y = 120.0
        # Trim vele continuo (0=lasco, 1=cazzato) con inerzia meccanica
        self.max_trim_delta_per_step = 0.10
        self.default_trim_level = 0.60
        # Inerzia velocità (momentum). Altissima sui foil (scivola nell'aria), bassa in acqua
        self.displacement_inertia = 0.85
        self.foiling_inertia = 0.98

        # Foiling characteristics
        self.foiling_takeoff_speed = 18.0  # Takeoff threshold as requested (18 kts)
        self.foiling_drop_speed = 15.0     # Hysteresis for dropping off the foil (15 kts)

        # Collision model (phase 2)
        self.collision_radius = 20.0
        self.near_collision_radius = 40.0
        self.max_collision_correction = 6.0
        self.collision_penalty_scale = 20.0
        self.near_collision_penalty_scale = 3.0
        # Regola base precedenze: su mure opposte, la barca su mura sinistra paga di più.
        self.port_tack_collision_multiplier = 1.6
        # Penalita' predittiva: rischio collisione nei prossimi secondi (time-to-collision)
        self.ttc_horizon = 4.0
        self.ttc_penalty_scale = 5.0
        # Penalita' "infinita" pratica: valore molto alto ma stabile numericamente per PPO.
        self.hard_violation_penalty = 10_000.0
        # Le regole hard partono dopo warmup per consentire apprendimento iniziale.
        self.hard_rules_warmup_steps = 20_000
        self.boundary_hard_after_steps = 15
        # Spin detection (giro completo + progresso scarso): comportamento irrealistico in regata.
        self.spin_window_len = 30
        self.spin_turn_threshold = np.radians(450.0)
        self.spin_min_progress = 0.0
        # Rounding robustness: evita loop improduttivi durante il giro boa.
        self.rounding_grace_steps = 20
        self.rounding_step_penalty_scale = 0.35
        self.rounding_step_penalty_cap = 20.0
        self.rounding_timeout_steps = 120
        self.rounding_retry_penalty = 350.0
        self.rounding_max_retries = 2

        # --- Costanti del Campo di Regata (Windward-Leeward) ---
        self.course_center_x = self.field_width / 2.0
        self.boundaries = {'x_min': 0.0, 'x_max': self.field_width}
        self.top_gate_y = self.field_length - 200.0
        self.bottom_gate_y = 200.0
        self.gate_width = 300.0

        # Obs: (x, y, speed, sin_h, cos_h, sin_aw, cos_aw, wind_speed, dist, sin_rb, cos_rb,
        #       rudder, sail_trim, is_foiling, active_foil, is_upwind_leg,
        #       rel_opp_x, rel_opp_y, opp_dist, speed_adv)
        self.observation_spaces = {agent: spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=np.float32
        ) for agent in self.possible_agents}

        self.action_spaces = {agent: spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        ) for agent in self.possible_agents}
        
        self.state = {}
        self.target = {}
        self.wind_field = WindField(field_size=int(max(self.field_width, self.field_length)))
        self.step_count = 0
        self.trajectory = {}
        self.previous_distance = {}
        self.best_distance = {}
        self.round_marks = {}
        self.fig = None
        self.ax = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_polar_speed(self, apparent_wind_angle, wind_speed, is_foiling, sail_trim):
        # NOTA: apparent_wind_angle e' in realtà il True Wind Angle (TWA)!
        angle_deg = normalize_twa_deg(apparent_wind_angle)
            
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
            
        base_speed = min(speed_ratio * wind_speed, self.max_speed)
        optimal_trim = optimal_trim_for_twa(angle_deg, is_foiling)
        trim_eff = trim_efficiency(sail_trim, optimal_trim, is_foiling)
        speed = min(base_speed * trim_speed_multiplier(trim_eff, is_foiling), self.max_speed)
        return speed, trim_eff, optimal_trim, angle_deg

    def _normalize_angle(self, angle):
        return angle % (2 * np.pi)

    def _compute_vmg_to_target(self, agent: str) -> float:
        """Velocity made good (kts) lungo la direzione del target corrente."""
        pos = np.array([self.state[agent]['x'], self.state[agent]['y']], dtype=np.float32)
        target_vec = self.target[agent].astype(np.float32) - pos
        target_norm = float(np.linalg.norm(target_vec))
        if target_norm < 1e-6:
            return 0.0

        target_unit = target_vec / target_norm
        heading = self.state[agent]['heading']
        speed = float(self.state[agent]['speed'])
        boat_vel = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32) * speed
        return float(np.dot(boat_vel, target_unit))

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
                'sail_trim': self.default_trim_level,
                'is_foiling': False,
                'active_foil': 1.0,
                'dropped_foil_penalty_applied': False,
                'current_leg': 1, # 1: Bolina, 2: Poppa, 3: Giro finale attorno boa arrivo
                'post_round_pending': False,
                'spin_turn_window': [],
                'spin_progress_window': [],
                'boundary_outside_steps': 0,
                'rounding_steps': 0,
                'rounding_retries': 0,
                'rounding_segment': None,
                'rounding_side': 0,
            }

            # Ogni barca deve girare realmente una boa (sinistra/destra) e non solo tagliare la linea gate.
            rounding_side = float(self.np_random.choice([-1, 1]))
            round_mark_x = self.course_center_x + rounding_side * (self.gate_width / 2.0)
            self.round_marks[agent] = {'side': rounding_side, 'x': round_mark_x}
            
            # Bersaglio Iniziale: boa da girare (non il centro gate).
            self.target[agent] = np.array([round_mark_x, self.top_gate_y])
            
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

            local_wind_dir, _ = self.wind_field.get_local_wind(
                self.state[agent]['x'], self.state[agent]['y']
            )
            twa = (local_wind_dir + np.pi) - self.state[agent]['heading']
            twa_deg = normalize_twa_deg(twa)
            self.state[agent]['sail_trim'] = optimal_trim_for_twa(twa_deg, False)

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        return observations, infos

    def _set_rounding_target(self, agent: str, gate_left: float, gate_right: float, gate_y: float, side: float, down_offset: float):
        """Imposta il target esterno alla boa lato scelto per completare il rounding."""
        ext_offset = 60.0
        if side < 0:
            self.target[agent] = np.array([gate_left - ext_offset, gate_y + down_offset])
        else:
            self.target[agent] = np.array([gate_right + ext_offset, gate_y + down_offset])

    def _start_rounding_segment(self, agent: str, segment: str, side: float):
        self.state[agent]['post_round_pending'] = True
        self.state[agent]['rounding_steps'] = 0
        self.state[agent]['rounding_retries'] = 0
        self.state[agent]['rounding_segment'] = segment
        self.state[agent]['rounding_side'] = float(side)

    def _apply_rounding_control(self, agent: str, reward: float, terminated: bool, hard_rules_active: bool, gate_left: float, gate_right: float):
        """Penalita' progressiva e retry per rounding non completato."""
        if terminated:
            return reward, terminated, 0.0

        if not self.state[agent].get('post_round_pending', False):
            self.state[agent]['rounding_steps'] = 0
            self.state[agent]['rounding_retries'] = 0
            self.state[agent]['rounding_segment'] = None
            return reward, terminated, 0.0

        self.state[agent]['rounding_steps'] = self.state[agent].get('rounding_steps', 0) + 1
        step_count = self.state[agent]['rounding_steps']
        over_steps = max(0, step_count - self.rounding_grace_steps)
        round_pen = min(self.rounding_step_penalty_cap, over_steps * self.rounding_step_penalty_scale)
        reward -= round_pen

        if step_count <= self.rounding_timeout_steps:
            return reward, terminated, round_pen

        self.state[agent]['rounding_retries'] = self.state[agent].get('rounding_retries', 0) + 1
        retries = self.state[agent]['rounding_retries']
        reward -= self.rounding_retry_penalty * min(3, retries)

        # Se insiste troppo e le hard rules sono attive: termina con violazione dedicata.
        if hard_rules_active and retries >= self.rounding_max_retries:
            reward -= self.hard_violation_penalty
            terminated = True
            self.state[agent]['termination_reason'] = 'failed_rounding'
            return reward, terminated, round_pen

        # Retry del rounding: reset timer e riporta target esterno alla boa corretta.
        self.state[agent]['rounding_steps'] = 0
        side = float(self.state[agent].get('rounding_side', 1.0))
        segment = self.state[agent].get('rounding_segment', None)
        if segment == 'top_to_bottom':
            self._set_rounding_target(agent, gate_left, gate_right, self.top_gate_y, side, down_offset=-40.0)
        elif segment == 'bottom_finish':
            self._set_rounding_target(agent, gate_left, gate_right, self.bottom_gate_y, side, down_offset=40.0)

        return reward, terminated, round_pen

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
        
        opponent = next((a for a in self.possible_agents if a != agent), None)
        if opponent is not None and opponent in self.state:
            opp_pos = np.array([self.state[opponent]['x'], self.state[opponent]['y']], dtype=np.float32)
            rel_vec = np.array([
                (opp_pos[0] - pos[0]) / self.field_width,
                (opp_pos[1] - pos[1]) / self.field_length,
            ], dtype=np.float32)
            opp_dist_norm = float(np.linalg.norm(opp_pos - pos) / self.field_diag)
            speed_adv = float((self.state[agent]['speed'] - self.state[opponent]['speed']) / self.max_speed)
            rel_opp_x = float(np.clip(rel_vec[0], -1.0, 1.0))
            rel_opp_y = float(np.clip(rel_vec[1], -1.0, 1.0))
            opp_dist_norm = float(np.clip(opp_dist_norm, 0.0, 1.0))
            speed_adv = float(np.clip(speed_adv, -1.0, 1.0))
        else:
            rel_opp_x = 0.0
            rel_opp_y = 0.0
            opp_dist_norm = 1.0
            speed_adv = 0.0

        obs = np.array([
            self.state[agent]['x'] / self.field_width,           # [0, 1]
            self.state[agent]['y'] / self.field_length,          # [0, 1]
            self.state[agent]['speed'] / self.max_speed,         # [0, 1]
            np.sin(heading), np.cos(heading),                    # [-1, 1]
            np.sin(apparent_wind), np.cos(apparent_wind),        # [-1, 1]
            local_wind_speed / self.max_wind,                    # [0, 1]
            dist_to_target / self.field_diag,                    # [0, 1]
            np.sin(rel_bearing), np.cos(rel_bearing),            # [-1, 1]
            float(self.state[agent]['rudder_angle']),             # [-1, 1]
            trim_level_to_action(self.state[agent]['sail_trim']), # [-1, 1]
            1.0 if self.state[agent]['is_foiling'] else 0.0,     # [0, 1] boolean foiling state
            self.state[agent]['active_foil'],                    # [-1, 1] Port vs Starboard foil
            1.0 if self.state[agent]['current_leg'] == 1 else -1.0, # [-1, 1] Leg information
            rel_opp_x,                                            # [-1, 1]
            rel_opp_y,                                            # [-1, 1]
            opp_dist_norm,                                        # [0, 1]
            speed_adv,                                            # [-1, 1]
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
        processed_agents = []
        collision_radius = self.collision_radius
        near_collision_radius = self.near_collision_radius
        max_correction_per_boat = self.max_collision_correction
        hard_rules_active = self.step_count >= self.hard_rules_warmup_steps

        for agent, action in actions.items():
            # --- salta agenti già terminati ---
            if agent not in self.agents:
                continue

            pos = np.array([self.state[agent]['x'], self.state[agent]['y']])
            prev_dist = np.linalg.norm(pos - self.target[agent])
            prev_y = float(self.state[agent]['y'])
            
            prev_rudder = self.state[agent].get('rudder_angle', 0.0)
            prev_trim = self.state[agent].get('sail_trim', self.default_trim_level)
            prev_heading = float(self.state[agent]['heading'])

            local_wind_dir, local_wind_speed = self.wind_field.get_local_wind(
                self.state[agent]['x'], self.state[agent]['y']
            )

            rudder_raw = action[0] if hasattr(action, '__len__') else action
            trim_raw = action[1] if (hasattr(action, '__len__') and len(action) > 1) else None

            rudder_input = float(np.clip(rudder_raw, -1.0, 1.0))
            self.state[agent]['rudder_angle'] = rudder_input

            # calcola heading e velocità
            speed_factor = self.state[agent]['speed'] / self.max_speed
            effective_factor = self.min_turn_factor + (1.0 - self.min_turn_factor) * speed_factor
            turn_rate = rudder_input * effective_factor * self.max_turn_per_step
            self.state[agent]['heading'] = self._normalize_angle(
                self.state[agent]['heading'] + turn_rate * self.dt
            )

            heading_delta = self._normalize_angle(self.state[agent]['heading'] - prev_heading)
            if heading_delta > np.pi:
                heading_delta -= 2 * np.pi
            spin_turn_window = self.state[agent].setdefault('spin_turn_window', [])
            spin_turn_window.append(abs(float(heading_delta)))
            if len(spin_turn_window) > self.spin_window_len:
                spin_turn_window.pop(0)

            apparent_wind_angle = (local_wind_dir + np.pi) - self.state[agent]['heading']
            twa_deg = normalize_twa_deg(apparent_wind_angle)

            if trim_raw is None:
                # Compatibilità con policy legacy (azione 1D): usa auto-trim verso l'ottimo.
                trim_target_level = optimal_trim_for_twa(twa_deg, self.state[agent]['is_foiling'])
            else:
                trim_target_level = action_to_trim_level(float(np.clip(trim_raw, -1.0, 1.0)))

            trim_delta = float(np.clip(
                trim_target_level - prev_trim,
                -self.max_trim_delta_per_step,
                self.max_trim_delta_per_step,
            ))
            self.state[agent]['sail_trim'] = float(np.clip(prev_trim + trim_delta, 0.0, 1.0))

            # 3. Frenata virata brusca: ridotta in foiling per permettere virate/strambate più vere e lunghe
            if self.state[agent]['is_foiling']:
                brake_factor = (abs(rudder_input) ** 1.5) * 0.05
                self.state[agent]['speed'] *= (1.0 - brake_factor)
            else:
                brake_factor = (abs(rudder_input) ** 1.5) * 0.35
                self.state[agent]['speed'] *= (1.0 - brake_factor)

            target_speed, trim_eff, trim_target, _ = self._get_polar_speed(
                apparent_wind_angle,
                local_wind_speed,
                self.state[agent]['is_foiling'],
                self.state[agent]['sail_trim'],
            )
            
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

            trim_target = optimal_trim_for_twa(twa_deg, self.state[agent]['is_foiling'])
            trim_eff = trim_efficiency(
                self.state[agent]['sail_trim'],
                trim_target,
                self.state[agent]['is_foiling'],
            )
            trim_error = abs(self.state[agent]['sail_trim'] - trim_target)
            vmg = self._compute_vmg_to_target(agent)
            vmg_norm = float(np.clip(vmg / self.max_speed, -1.0, 1.0))

            dropped_foil = was_foiling and not self.state[agent]['is_foiling']
            displacement = self.state[agent]['speed'] * 0.514 * self.dt
            self.state[agent]['x'] += displacement * np.cos(self.state[agent]['heading'])
            self.state[agent]['y'] += displacement * np.sin(self.state[agent]['heading'])

            self.trajectory[agent].append(np.array([self.state[agent]['x'], self.state[agent]['y']]))

            # Collision handling (single pass): confronta l'agente corrente con quelli gia' processati.
            # In questo modo evitiamo doppio conteggio e manteniamo la penalita' additiva sul reward finale.
            collision_penalty = 0.0
            collision_count = 0
            near_collision_penalty = 0.0
            ttc_penalty = 0.0
            ttc_risk_count = 0
            row_violation_count = 0
            hard_violation = False
            hard_violation_reason = None
            for other in processed_agents:
                if other not in self.state:
                    continue

                pos_curr = np.array([self.state[agent]['x'], self.state[agent]['y']], dtype=np.float32)
                pos_other = np.array([self.state[other]['x'], self.state[other]['y']], dtype=np.float32)
                delta = pos_curr - pos_other
                dist = float(np.linalg.norm(delta))

                if not (1e-6 < dist < near_collision_radius):
                    continue

                # Penalita' morbida di prossimita': evita ingaggi troppo ravvicinati prima del contatto.
                near_ratio = (near_collision_radius - dist) / near_collision_radius
                near_pen = near_ratio * self.near_collision_penalty_scale
                near_collision_penalty += near_pen
                rewards.setdefault(other, 0.0)
                rewards[other] -= near_pen
                if other in infos:
                    infos[other]['near_collision_penalty'] = infos[other].get('near_collision_penalty', 0.0) + near_pen

                # Penalita' predittiva TTC: se le traiettorie convergono rapidamente, paga gia' in anticipo.
                delta_hat = delta / dist
                vel_curr = np.array([
                    np.cos(self.state[agent]['heading']) * self.state[agent]['speed'],
                    np.sin(self.state[agent]['heading']) * self.state[agent]['speed'],
                ], dtype=np.float32)
                vel_other = np.array([
                    np.cos(self.state[other]['heading']) * self.state[other]['speed'],
                    np.sin(self.state[other]['heading']) * self.state[other]['speed'],
                ], dtype=np.float32)
                rel_vel = vel_curr - vel_other
                closing_speed = float(max(0.0, -np.dot(delta_hat, rel_vel)))
                if closing_speed > 1e-3:
                    ttc = dist / closing_speed
                    if 0.0 < ttc < self.ttc_horizon:
                        ttc_ratio = 1.0 - (ttc / self.ttc_horizon)
                        ttc_pen = ttc_ratio * near_ratio * self.ttc_penalty_scale
                        ttc_penalty += ttc_pen
                        ttc_risk_count += 1
                        rewards.setdefault(other, 0.0)
                        rewards[other] -= ttc_pen
                        if other in infos:
                            infos[other]['ttc_penalty'] = infos[other].get('ttc_penalty', 0.0) + ttc_pen
                            infos[other]['ttc_risk_count'] = infos[other].get('ttc_risk_count', 0) + 1

                if dist >= collision_radius:
                    continue

                overlap_ratio = (collision_radius - dist) / collision_radius
                base_penalty = overlap_ratio * self.collision_penalty_scale

                penalty_curr = base_penalty
                penalty_other = base_penalty

                # Precedenza semplificata: su mure opposte penalizza maggiormente chi e' su mura sinistra.
                opposite_tacks = (self.state[agent]['active_foil'] * self.state[other]['active_foil']) < 0.0
                if opposite_tacks:
                    curr_is_port = self.state[agent]['active_foil'] < 0.0
                    other_is_port = self.state[other]['active_foil'] < 0.0
                    if curr_is_port and not other_is_port:
                        penalty_curr *= self.port_tack_collision_multiplier
                        row_violation_count += 1
                    elif other_is_port and not curr_is_port:
                        penalty_other *= self.port_tack_collision_multiplier
                        if other in infos:
                            infos[other]['row_violation_count'] = infos[other].get('row_violation_count', 0) + 1

                collision_penalty += penalty_curr
                collision_count += 1

                # Collisione hard solo se impatto severo e dopo warmup.
                hard_collision = hard_rules_active and dist < (collision_radius * 0.6) and closing_speed > 6.0
                if hard_collision:
                    hard_violation = True
                    hard_violation_reason = 'collision'

                rewards.setdefault(other, 0.0)
                rewards[other] -= penalty_other
                if hard_collision:
                    rewards[other] -= self.hard_violation_penalty
                    terminations[other] = True
                    truncations[other] = False
                    if other in self.state:
                        self.state[other]['termination_reason'] = 'collision'

                # Piccola separazione simmetrica per evitare interpenetrazione persistente.
                overlap = collision_radius - dist
                corr_mag = min(max_correction_per_boat, overlap * 0.5)
                direction = delta / dist
                correction = direction * corr_mag

                self.state[agent]['x'] += float(correction[0])
                self.state[agent]['y'] += float(correction[1])
                self.state[other]['x'] -= float(correction[0])
                self.state[other]['y'] -= float(correction[1])

                # Aggiorna osservazione e info dell'agente gia' processato dopo la correzione posizione.
                if other in observations:
                    observations[other] = self._get_obs(other)
                if other in infos:
                    infos[other]['collision_penalty'] = infos[other].get('collision_penalty', 0.0) + penalty_other
                    infos[other]['collision_count'] = infos[other].get('collision_count', 0) + 1
                    if hard_collision:
                        infos[other]['termination_reason'] = 'collision'

            # distanza dal target
            pos = np.array([self.state[agent]['x'], self.state[agent]['y']])
            dist_to_target = np.linalg.norm(pos - self.target[agent])

            reward = 0.0
            terminated = False
            truncated = False
            # Reset terminal marker each step while agent is alive.
            self.state[agent]['termination_reason'] = None
            # 1. Progresso verso target e VMG (Velocity Made Good)
            distance_delta = prev_dist - dist_to_target
            spin_progress_window = self.state[agent].setdefault('spin_progress_window', [])
            spin_progress_window.append(float(distance_delta))
            if len(spin_progress_window) > self.spin_window_len:
                spin_progress_window.pop(0)

            # Progresso puro: premio forte quando riduce la distanza, punizione più netta quando si allontana.
            progress_gain = 1.8 if self.state[agent]['is_foiling'] else 0.65
            if distance_delta >= 0:
                reward += distance_delta * progress_gain
            else:
                reward += distance_delta * (progress_gain * 1.8)

            # Allineamento rotta-target: favorisce traiettorie pulite evitando zig-zag improduttivo.
            target_bearing = np.arctan2(self.target[agent][1] - pos[1], self.target[agent][0] - pos[0])
            heading_error = self._normalize_angle(target_bearing - self.state[agent]['heading'])
            if heading_error > np.pi:
                heading_error -= 2 * np.pi
            heading_error_norm = abs(heading_error) / np.pi
            reward += (1.0 - heading_error_norm) * 1.2
            if heading_error_norm > 0.75:
                reward -= (heading_error_norm - 0.75) * 3.0

            # VMG shaping esplicito per massimizzare velocità utile su bolina e poppa.
            leg_vmg_weight = 1.6 if self.state[agent]['current_leg'] == 1 else 1.25
            reward += max(vmg_norm, 0.0) * leg_vmg_weight * 8.5
            reward += min(vmg_norm, 0.0) * leg_vmg_weight * 4.0

            # Reward competitivo: piccolo bonus se il distacco dal target e' migliore dell'avversario.
            opponent = next((a for a in self.possible_agents if a != agent), None)
            if opponent is not None and opponent in self.state:
                opp_pos = np.array([self.state[opponent]['x'], self.state[opponent]['y']], dtype=np.float32)
                opp_dist_to_target = float(np.linalg.norm(opp_pos - self.target[opponent]))
                tactical_advantage = (opp_dist_to_target - dist_to_target) / self.field_diag
                reward += float(np.clip(tactical_advantage, -1.0, 1.0)) * 1.5

            # Shaping diretto per direzione di gamba sulla coordinata Y.
            # Leg 1 (bolina): salire verso top gate. Leg 2 (poppa): scendere verso bottom gate.
            leg_delta_y = float(self.state[agent]['y'] - prev_y)
            if self.state[agent]['current_leg'] == 1:
                reward += leg_delta_y * 0.15
            else:
                reward -= leg_delta_y * 0.15

            # 2. Costo per step (urgenza)
            reward -= 0.25

            # Penalita' collisioni accumulate durante lo step.
            reward -= collision_penalty
            reward -= near_collision_penalty
            reward -= ttc_penalty

            round_penalty = 0.0

            # 3. Penalità timone logaritmica/esponenziale (ridotta, affidiamoci all'attrito fisico)
            reward -= (abs(rudder_input) ** 3) * 2.0
            rudder_delta = rudder_input - prev_rudder
            reward -= (abs(rudder_delta) ** 2) * 5.0

            # 3.c Trim sails: premia assetto efficiente, penalizza movimenti bruschi e fuori-trim ad alta velocità
            reward += (trim_eff - 0.72) * 8.5
            reward -= (abs(trim_delta) ** 1.5) * 5.5
            leg_trim_threshold = 0.18 if self.state[agent]['current_leg'] == 1 else 0.24
            if self.state[agent]['speed'] > 10.0 and trim_error > leg_trim_threshold:
                reward -= (trim_error - leg_trim_threshold) * 18.0

            # Accoppia trim a VMG: trim corretto quando la barca accelera verso il target.
            reward += max(vmg_norm, 0.0) * (trim_eff - 0.50) * 5.0

            # 3.b Drop off foil penalty (ridotta: la vera penalità sarà chimica/fisica causata dalla perdita di VMG)
            if dropped_foil:
                reward -= 40.0  # Penalità tattica, ma non così alta da impedire le virate di layline

            # 4. Foiling and SPEED INCENTIVES (Primary focus of the agent)
            if self.state[agent]['is_foiling']:
                speed_over_threshold = self.state[agent]['speed'] - self.foiling_drop_speed
                reward += (speed_over_threshold * 1.1)
            else:
                reward -= 1.5
                
            if self.state[agent]['speed'] < self.foiling_drop_speed:
                reward -= 3.0  # Più pressione a recuperare velocità per tornare competitivo

            # Velocità utile per gamba: in bolina basta meno speed assoluta, in poppa target più alto.
            leg_target_speed = 20.0 if self.state[agent]['current_leg'] == 1 else 24.0
            speed_deficit = max(0.0, leg_target_speed - self.state[agent]['speed'])
            reward -= speed_deficit * 0.35
                
            # Bonus velocità assoluta quando ci si avvicina (secondary)
            if distance_delta > 0:
                reward += self.state[agent]['speed'] * 0.28

            if dist_to_target < self.best_distance[agent]:
                self.best_distance[agent] = dist_to_target

            # 6. Gate passing & Boundary logic
            bx = self.state[agent]['x']
            by = self.state[agent]['y']
            
            # Superamento linee tratteggiate: soft in apprendimento, hard solo se persistente dopo warmup.
            if bx < self.boundaries['x_min'] or bx > self.boundaries['x_max']:
                self.state[agent]['boundary_outside_steps'] = self.state[agent].get('boundary_outside_steps', 0) + 1
                reward -= 30.0
                if hard_rules_active and self.state[agent]['boundary_outside_steps'] >= self.boundary_hard_after_steps:
                    reward -= self.hard_violation_penalty
                    terminated = True
                    self.state[agent]['termination_reason'] = 'boundary_violation'
            else:
                self.state[agent]['boundary_outside_steps'] = 0
                
            # Fuori mappa totale: hard-fail.
            if bx < 0 or bx > self.field_width or by < 0 or by > self.field_length:
                reward -= self.hard_violation_penalty
                terminated = True
                self.state[agent]['termination_reason'] = 'out_of_bounds'

            # Controllo attraversamento Cancello (Gate)
            gate_left = self.course_center_x - self.gate_width / 2.0
            gate_right = self.course_center_x + self.gate_width / 2.0
            
            if self.state[agent]['current_leg'] == 1:
                # Upwind: deve tagliare in mezzo alle boe (Y >= top_gate_y) passando fra gate_left e gate_right.
                mark_info = self.round_marks[agent]
                mark_x = mark_info['x']
                mark_side = mark_info['side']

                crossed_top_line = prev_y < self.top_gate_y <= by
                # Se supera la linea della boa senza passare nel gate e' hard-fail (dopo warmup).
                if hard_rules_active and crossed_top_line and not (gate_left <= bx <= gate_right):
                    reward -= self.hard_violation_penalty
                    terminated = True
                    self.state[agent]['termination_reason'] = 'missed_top_gate'
                
                # Check del taglio corretto della linea fra le due boe:
                if by >= self.top_gate_y and gate_left <= bx <= gate_right:
                    self.state[agent]['current_leg'] = 2
                    reward += 500.0  # Super Bonus per aver girato la boa di bolina
                    round_side = -1.0 if bx < self.course_center_x else 1.0
                    self._start_rounding_segment(agent, segment='top_to_bottom', side=round_side)
                    # Ora imposta il target esterno alla boa per costringere a fare il giro attorno a essa
                    self._set_rounding_target(agent, gate_left, gate_right, self.top_gate_y, round_side, down_offset=-40.0)

                    self.previous_distance[agent] = np.linalg.norm(pos - self.target[agent])
                    self.best_distance[agent] = self.previous_distance[agent]
                    
            elif self.state[agent]['current_leg'] == 2:
                # Prima di puntare il gate di arrivo, obbliga un'uscita pulita dalla boa.
                if self.state[agent].get('post_round_pending', False):
                    if dist_to_target <= max(self.target_radius * 1.2, 60.0):
                        self.state[agent]['post_round_pending'] = False
                        self.state[agent]['rounding_segment'] = None
                        self.state[agent]['rounding_steps'] = 0
                        self.state[agent]['rounding_retries'] = 0
                        self.target[agent] = np.array([self.course_center_x, self.bottom_gate_y])
                        self.previous_distance[agent] = np.linalg.norm(pos - self.target[agent])
                        self.best_distance[agent] = self.previous_distance[agent]

                # Downwind: deve scendere sotto la Y del Bottom Gate, restando in mezzo alle boe X.
                # Al passaggio del Bottom Gate, attiva Leg 3 (giro finale attorno boa).
                if (not self.state[agent].get('post_round_pending', False)) and by <= self.bottom_gate_y and gate_left <= bx <= gate_right:
                    self.state[agent]['current_leg'] = 3
                    reward += 500.0  # Bonus per aver girato la boa di arrivo
                    round_side = -1.0 if bx < self.course_center_x else 1.0
                    self._start_rounding_segment(agent, segment='bottom_finish', side=round_side)
                    # Imposta target esterno alla boa di arrivo per costringere il giro finale
                    self._set_rounding_target(agent, gate_left, gate_right, self.bottom_gate_y, round_side, down_offset=40.0)
                    
                    self.previous_distance[agent] = np.linalg.norm(pos - self.target[agent])
                    self.best_distance[agent] = self.previous_distance[agent]
            
            elif self.state[agent]['current_leg'] == 3:
                # Leg finale: giro attorno boa di arrivo. Una volta raggiunto il target esterno, contrassegna la fine.
                if self.state[agent].get('post_round_pending', False):
                    if dist_to_target <= max(self.target_radius * 1.2, 60.0):
                        efficiency = max(0, self.max_steps - self.step_count) / self.max_steps
                        reward += 2000.0 + efficiency * 1000.0  # Vittoria finale della regata
                        terminated = True
                        self.state[agent]['steps_to_target'] = self.step_count
                        self.state[agent]['termination_reason'] = 'finished_race'
                        self.state[agent]['post_round_pending'] = False
                        self.state[agent]['rounding_segment'] = None
                        self.state[agent]['rounding_steps'] = 0
                        self.state[agent]['rounding_retries'] = 0

            # Protezione rounding: penalita' crescente + retry + hard-fail dedicato.
            reward, terminated, round_penalty = self._apply_rounding_control(
                agent=agent,
                reward=reward,
                terminated=terminated,
                hard_rules_active=hard_rules_active,
                gate_left=gate_left,
                gate_right=gate_right,
            )

            if self.step_count >= self.max_steps:
                truncated = True
                self.state[agent]['termination_reason'] = 'timeout'
                progress = 1.0 - (self.best_distance[agent] / max(self.previous_distance[agent], 1.0))
                if progress > 0:
                    reward += progress * 200.0

            # Giro su se stessa + progresso scarso/negativo: hard-fail.
            # ESCLUSIONE: non penalizzare durante Leg 3 (giro finale è voluto attorno boa di arrivo).
            if hard_rules_active and not terminated and self.state[agent]['current_leg'] != 3:
                turn_sum = float(np.sum(self.state[agent].get('spin_turn_window', [])))
                prog_sum = float(np.sum(self.state[agent].get('spin_progress_window', [])))
                if len(self.state[agent].get('spin_turn_window', [])) >= self.spin_window_len and turn_sum >= self.spin_turn_threshold and prog_sum <= self.spin_min_progress:
                    reward -= self.hard_violation_penalty
                    terminated = True
                    self.state[agent]['termination_reason'] = 'spin_violation'

            # Hard violation da collisione rilevata nel blocco pairwise.
            if hard_violation and not terminated:
                reward -= self.hard_violation_penalty
                terminated = True
                self.state[agent]['termination_reason'] = hard_violation_reason

            termination_reason = self.state[agent].get('termination_reason', None)
            finished_race = termination_reason == 'finished_race'

            # aggiorna dizionari
            observations[agent] = self._get_obs(agent)
            rewards[agent] = reward
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = {
                'agent': agent,
                'distance_to_target': dist_to_target,
                'speed': self.state[agent]['speed'],
                'trim': self.state[agent]['sail_trim'],
                'trim_efficiency': trim_eff,
                'trim_target': trim_target,
                'trim_error': trim_error,
                'vmg': vmg,
                'vmg_norm': vmg_norm,
                'leg': self.state[agent]['current_leg'],
                'steps': self.step_count,
                'steps_to_target': self.state[agent].get('steps_to_target', None),
                'best_distance': self.best_distance[agent],
                'finished_race': finished_race,
                'termination_reason': termination_reason,
                'terminated': terminated,
                'truncated': truncated,
                'collision_penalty': collision_penalty,
                'collision_count': collision_count,
                'near_collision_penalty': near_collision_penalty,
                'ttc_penalty': ttc_penalty,
                'ttc_risk_count': ttc_risk_count,
                'row_violation_count': row_violation_count,
                'rounding_penalty': round_penalty,
                'rounding_steps': self.state[agent].get('rounding_steps', 0),
                'rounding_retries': self.state[agent].get('rounding_retries', 0),
                'rounding_segment': self.state[agent].get('rounding_segment', None),
            }

            processed_agents.append(agent)

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
        self.ax.set_xlim(0, self.field_width)
        self.ax.set_ylim(0, self.field_length)
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
        self.ax.plot([self.boundaries['x_min'], self.boundaries['x_min']], [0, self.field_length], 'r--', linewidth=2, alpha=0.5, label='Boundary')
        self.ax.plot([self.boundaries['x_max'], self.boundaries['x_max']], [0, self.field_length], 'r--', linewidth=2, alpha=0.5)

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

                # Overlay diagnostico collisioni (raggio contatto e near-collision)
                near_circle = patches.Circle(
                    (bx, by),
                    self.near_collision_radius,
                    fill=False,
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.18,
                    edgecolor='orange',
                )
                coll_circle = patches.Circle(
                    (bx, by),
                    self.collision_radius,
                    fill=False,
                    linestyle='-',
                    linewidth=1.0,
                    alpha=0.25,
                    edgecolor='red',
                )
                self.ax.add_patch(near_circle)
                self.ax.add_patch(coll_circle)
                
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

                trim_percent = int(self.state[agent].get('sail_trim', self.default_trim_level) * 100)
                self.ax.text(bx + 18, by + 7,
                    f"Trim:{trim_percent:02d}%",
                    fontsize=7, color='navy', fontweight='bold')
        
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
                f"Speed: {self.state[ref_agent]['speed']:.1f} kts | "
                f"Trim: {self.state[ref_agent].get('sail_trim', self.default_trim_level) * 100:.0f}% | "
                f"Dist Y to Gate: {dist_to_gate:.0f}m",
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