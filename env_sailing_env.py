"""
Modulo: env_sailing_env
=======================
Gestisce lo stato dell'ambiente Multi-Agent (PettingZoo ParallelEnv), applica le azioni,
calcola le penalità/reward ad ogni step e agisce da ponte tra i layer fisici e i layer grafici.

Architettura:
- Usa `core_wind_model.py` per modellare il vento sull'arena.
- Usa `core_boat_physics.py` e `core_sail_trim.py` per far evolvere lo stato cinematico delle barche.
- Usa `env_rendering.py` per le chiamate visive esterne.

Note Tecniche Locali (Riassunte per RL):
- VMG (Velocity Made Good): Il reward principale modella esplicitamente l'avvicinamento
  Vero (y-progress per le legate e VMG spaziale), disincentivando il semplice andare veloci
  nella direzione sbagliata.
- Foiling: L'IA è fortemente disincentivata dalla caduta dai foil ("dropped_foil_penalty"),
  ed è incoraggiata primariamente a mantenere alte velocità una volta decollata.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from typing import Optional

# Componenti architetturali esterne refattorizzate
from core_wind_model import WindField
from core_sail_trim import (
    action_to_trim_level,
    normalize_twa_deg,
    optimal_trim_for_twa,
    trim_level_to_action,
)
from core_boat_physics import compute_polar_speed, compute_vmg_to_target
from env_rendering import SailingRenderer
import yaml


class ImprovedSailingEnv(ParallelEnv):
    """
    Ambiente di navigazione a vela migliorato con due barche, compatibile con PettingZoo Parallel API.
    """

    metadata = {"render_modes": ["rgb_array"], "name": "sailing_v0"}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
        except Exception:
            config = {}
        phys_cfg = config.get("physics", {})

        # Geometria e Fisica Base
        self.field_length = 4075.0
        self.field_width = 1482.0
        self.max_speed = 50.0  # Incrementata per foiling
        self.max_wind = 30.0
        self.target_radius = 50.0  # Piccolo per precisione
        self.dt = 1.0
        self.max_steps = 1800
        self.render_mode = render_mode

        # Configurazione Multi-Agent
        self.possible_agents = ["red_boat", "blue_boat"]
        self.agents = self.possible_agents[:]

        # Timone ed Effetti Continuistici
        self.max_turn_per_step = np.radians(25)
        self.min_turn_factor = 0.12
        self.max_rudder_delta_per_step = phys_cfg.get("max_rudder_delta_per_step", 0.25)  # Max variazione timone per step
        self.mark_round_margin = 25.0
        self.post_round_offset_x = 90.0
        self.post_round_offset_y = 120.0

        # Trim vele continuo (0=lasco, 1=cazzato)
        self.max_trim_delta_per_step = 0.10
        self.default_trim_level = 0.60

        # Inerzia velocità (Momentum)
        # Altissima sui foil (scivola nell'aria limitando l'attrito), bassa in
        # acqua piena.
        self.displacement_inertia = phys_cfg.get("displacement_inertia", 0.85)
        self.foiling_inertia = phys_cfg.get("foiling_inertia", 0.98)

        # Foiling characteristics (soglie di decollo e ammaraggio con isteresi)
        self.foiling_takeoff_speed = 18.0
        self.foiling_drop_speed = 15.0

        # Costanti del Campo di Regata (Windward-Leeward)
        self.course_center_x = self.field_width / 2.0
        self.boundaries = {"x_min": 0.0, "x_max": self.field_width}
        self.top_gate_y = self.field_length - 200.0
        self.bottom_gate_y = 200.0
        self.gate_width = 300.0

        # Spazi Osservazione / Azione
        # Obs: (x, y, speed, sin_h, cos_h, sin_aw, cos_aw, wind_speed, dist, sin_rb, cos_rb,
        #       rudder, sail_trim, is_foiling, active_foil, leg_phase, time_remaining)
        self.observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Stato globale e moduli esterni
        self.state = {}
        self.target = {}
        self.wind_field = WindField(
            field_size=int(max(self.field_width, self.field_length))
        )
        self.renderer = SailingRenderer(self)
        self.step_count = 0
        self.trajectory = {}
        self.previous_distance = {}
        self.best_distance = {}
        self.round_marks = {}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _normalize_angle(self, angle: float) -> float:
        """Normalizza l'angolo tra 0 e 2*Pi."""
        return float(angle % (2 * np.pi))

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict, dict]:
        """
        Inizializza l'ambiente e ripristina la regata dall'inizio (Leg 1: Bolina).
        """
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        elif not hasattr(self, "np_random"):
            self.np_random, _ = gym.utils.seeding.np_random()

        self.agents = self.possible_agents[:]
        self.step_count = 0

        # Inizializziamo il vento discendente (Asse Y verso il basso)
        if options and "wind_direction" in options:
            base_dir = float(options["wind_direction"])
        else:
            base_dir = 1.5 * np.pi
        self.wind_field.reset(self.np_random, base_direction=base_dir)

        # Posizionamento scafi alle boe del gate di partenza
        # Le barche partono alla stessa y delle boe ma più in basso,
        # una alla x della boa sinistra e l'altra alla x della boa destra.
        gate_left_x = self.course_center_x - self.gate_width / 2.0
        gate_right_x = self.course_center_x + self.gate_width / 2.0
        start_y = self.bottom_gate_y - 80.0  # Sotto la linea di partenza

        # Assegnazione casuale: chi va a sinistra e chi a destra
        if self.np_random.random() < 0.5:
            side_assignment = [gate_left_x, gate_right_x]  # red sx, blue dx
        else:
            side_assignment = [gate_right_x, gate_left_x]  # red dx, blue sx

        for i, agent in enumerate(self.possible_agents):
            start_x = side_assignment[i]

            self.state[agent] = {
                "x": start_x,
                "y": start_y,
                "speed": 0.0,
                "heading": 0.0,  # verrà inizializzato tra poco
                "rudder_angle": 0.0,
                "sail_trim": self.default_trim_level,
                "is_foiling": False,
                "active_foil": 1.0,
                "dropped_foil_penalty_applied": False,
                "current_leg": 0,  # 0: Partenza, 1: Bolina, 2: Poppa
                "post_round_pending": False,
                "max_y_reached": start_y,
                "min_y_reached": start_y,
            }

            # Selezione tattica boa (sinistra/destra del top gate da puntare a Leg 1)
            rounding_side = float(self.np_random.choice([-1, 1]))
            round_mark_x = self.course_center_x + rounding_side * (
                self.gate_width / 2.0
            )
            self.round_marks[agent] = {"side": rounding_side, "x": round_mark_x}

            # Target iniziale: attraversamento del gate di partenza (Leg 0)
            self.target[agent] = np.array([self.course_center_x, self.bottom_gate_y])
            self.trajectory[agent] = [
                np.array([self.state[agent]["x"], self.state[agent]["y"]])
            ]

            pos = np.array([self.state[agent]["x"], self.state[agent]["y"]])
            self.previous_distance[agent] = float(
                np.linalg.norm(pos - self.target[agent])
            )
            self.best_distance[agent] = self.previous_distance[agent]

        # Allineamento iniziale prua (controvento / di bolina) a mure casuali
        for agent in self.possible_agents:
            tack_sign = float(self.np_random.choice([-1, 1]))
            start_heading = base_dir + np.pi + tack_sign * np.radians(50.0)
            start_heading += self.np_random.uniform(-np.radians(10), np.radians(10))
            self.state[agent]["heading"] = self._normalize_angle(start_heading)

            # Sincronizza il trim visuale iniziale
            local_wind_dir, _ = self.wind_field.get_local_wind(
                self.state[agent]["x"], self.state[agent]["y"]
            )
            twa = (local_wind_dir + np.pi) - self.state[agent]["heading"]
            twa_deg = normalize_twa_deg(twa)
            self.state[agent]["sail_trim"] = optimal_trim_for_twa(twa_deg, False)

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}

        return observations, infos

    def _get_obs(self, agent: str) -> np.ndarray:
        """Costruisce il vettore di osservazione di un singolo agente."""
        pos = np.array([self.state[agent]["x"], self.state[agent]["y"]])
        dist_to_target = float(np.linalg.norm(pos - self.target[agent]))
        bearing_to_target = np.arctan2(
            self.target[agent][1] - pos[1], self.target[agent][0] - pos[0]
        )

        local_wind_dir, local_wind_speed = self.wind_field.get_local_wind(
            self.state[agent]["x"], self.state[agent]["y"]
        )

        heading = self.state[agent]["heading"]
        rel_bearing = bearing_to_target - heading

        # Vento apparente mappato [-pi, +pi] per reti neurali simmetriche
        apparent_wind = (local_wind_dir + np.pi) - heading
        apparent_wind = self._normalize_angle(apparent_wind)
        if apparent_wind > np.pi:
            apparent_wind -= 2 * np.pi

        obs = np.array(
            [
                self.state[agent]["x"] / self.field_width,
                self.state[agent]["y"] / self.field_length,
                self.state[agent]["speed"] / self.max_speed,
                np.sin(heading),
                np.cos(heading),
                np.sin(apparent_wind),
                np.cos(apparent_wind),
                local_wind_speed / self.max_wind,
                dist_to_target / (np.sqrt(self.field_width**2 + self.field_length**2)),
                np.sin(rel_bearing),
                np.cos(rel_bearing),
                float(self.state[agent]["rudder_angle"]),
                trim_level_to_action(self.state[agent]["sail_trim"]),
                1.0 if self.state[agent]["is_foiling"] else 0.0,
                self.state[agent]["active_foil"],
                self.state[agent]["current_leg"] / 2.0,  # Leg: 0.0=partenza, 0.5=bolina, 1.0=poppa
                1.0 - (self.step_count / self.max_steps),  # Tempo rimanente [1→0]
            ],
            dtype=np.float32,
        )

        return np.clip(obs, -1.0, 1.0)

    def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]:
        """
        Esegue un passo di simulazione, applicando la cinematica,
        controllando la validità geografica, testando collisioni multiship
        e calcolando lo shaping del reward finale in un'unica cascata logica.
        """
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
            if agent not in self.agents:
                continue

            pos = np.array([self.state[agent]["x"], self.state[agent]["y"]])
            prev_dist = float(np.linalg.norm(pos - self.target[agent]))
            prev_y = float(self.state[agent]["y"])

            prev_rudder = self.state[agent].get("rudder_angle", 0.0)
            prev_trim = self.state[agent].get("sail_trim", self.default_trim_level)

            local_wind_dir, local_wind_speed = self.wind_field.get_local_wind(
                self.state[agent]["x"], self.state[agent]["y"]
            )

            # Estrazione azioni bi-dimensionali
            rudder_raw = action[0] if hasattr(action, "__len__") else action
            trim_raw = (
                action[1] if (hasattr(action, "__len__") and len(action) > 1) else None
            )

            rudder_target = float(np.clip(rudder_raw, -1.0, 1.0))
            # Limita la velocità di virata (Timone idraulico graduale)
            rudder_delta = np.clip(
                rudder_target - prev_rudder,
                -self.max_rudder_delta_per_step,
                self.max_rudder_delta_per_step
            )
            rudder_input = float(np.clip(prev_rudder + rudder_delta, -1.0, 1.0))
            self.state[agent]["rudder_angle"] = rudder_input

            # -----------------------------------------------------------------
            # [1] Dinamica Rotazionale (Timone Inerziale)
            # -----------------------------------------------------------------
            speed_factor = self.state[agent]["speed"] / self.max_speed
            effective_factor = (
                self.min_turn_factor + (1.0 - self.min_turn_factor) * speed_factor
            )
            turn_rate = rudder_input * effective_factor * self.max_turn_per_step
            self.state[agent]["heading"] = self._normalize_angle(
                self.state[agent]["heading"] + turn_rate * self.dt
            )

            apparent_wind_angle = (local_wind_dir + np.pi) - self.state[agent][
                "heading"
            ]
            twa_deg = normalize_twa_deg(apparent_wind_angle)

            # -----------------------------------------------------------------
            # [2] Dinamica Vele e Frenata Virata
            # -----------------------------------------------------------------
            if trim_raw is None:  # Retrocompatibilità
                trim_target_level = optimal_trim_for_twa(
                    twa_deg, self.state[agent]["is_foiling"]
                )
            else:
                trim_target_level = action_to_trim_level(
                    float(np.clip(trim_raw, -1.0, 1.0))
                )

            trim_delta = float(
                np.clip(
                    trim_target_level - prev_trim,
                    -self.max_trim_delta_per_step,
                    self.max_trim_delta_per_step,
                )
            )
            self.state[agent]["sail_trim"] = float(
                np.clip(prev_trim + trim_delta, 0.0, 1.0)
            )

            # Nota Tecnica: Frenata per virata brusca modificata (più clemente
            # in foiling per attrito inferiore)
            if self.state[agent]["is_foiling"]:
                brake_factor = (abs(rudder_input) ** 1.5) * 0.05
                self.state[agent]["speed"] *= 1.0 - brake_factor
            else:
                brake_factor = (abs(rudder_input) ** 1.5) * 0.35
                self.state[agent]["speed"] *= 1.0 - brake_factor

            # -----------------------------------------------------------------
            # [3] Applicazione Polar e Foiling (Moduli Core Esterni)
            # -----------------------------------------------------------------
            target_speed, trim_eff, trim_target, _ = compute_polar_speed(
                apparent_wind_angle=apparent_wind_angle,
                wind_speed=local_wind_speed,
                is_foiling=self.state[agent]["is_foiling"],
                sail_trim=self.state[agent]["sail_trim"],
                max_speed=self.max_speed,
            )

            # Smooth inerziale sulla velocità pura
            current_inertia = (
                self.foiling_inertia
                if self.state[agent]["is_foiling"]
                else self.displacement_inertia
            )
            self.state[agent]["speed"] = self.state[agent][
                "speed"
            ] * current_inertia + target_speed * (1.0 - current_inertia)

            # Logica Foiling Hysteresis
            was_foiling = self.state[agent]["is_foiling"]
            aw_norm = self._normalize_angle(apparent_wind_angle)
            if aw_norm > np.pi:
                aw_norm -= 2 * np.pi
            self.state[agent]["active_foil"] = 1.0 if aw_norm > 0 else -1.0

            if self.state[agent]["speed"] >= self.foiling_takeoff_speed:
                self.state[agent]["is_foiling"] = True
            elif self.state[agent]["speed"] < self.foiling_drop_speed:
                self.state[agent]["is_foiling"] = False

            trim_target = optimal_trim_for_twa(twa_deg, self.state[agent]["is_foiling"])
            trim_error = abs(self.state[agent]["sail_trim"] - trim_target)

            vmg = compute_vmg_to_target(
                boat_x=self.state[agent]["x"],
                boat_y=self.state[agent]["y"],
                heading=self.state[agent]["heading"],
                speed=self.state[agent]["speed"],
                target_x=self.target[agent][0],
                target_y=self.target[agent][1],
            )
            vmg_norm = float(np.clip(vmg / self.max_speed, -1.0, 1.0))

            dropped_foil = was_foiling and not self.state[agent]["is_foiling"]

            # Movimento spaziale
            displacement = (
                self.state[agent]["speed"] * 0.514 * self.dt
            )  # nts -> m/s proxy ratio
            self.state[agent]["x"] += displacement * np.cos(
                self.state[agent]["heading"]
            )
            self.state[agent]["y"] += displacement * np.sin(
                self.state[agent]["heading"]
            )

            self.trajectory[agent].append(
                np.array([self.state[agent]["x"], self.state[agent]["y"]])
            )

            reward = rewards.get(agent, 0.0)
            terminated = False
            truncated = False
            self.state[agent]["termination_reason"] = None

            # Le collisioni vengono calcolate globalmente a fine step.

            # -----------------------------------------------------------------
            # [5] Reward Shaping (Motori di spinta per la policy)
            # -----------------------------------------------------------------
            pos = np.array([self.state[agent]["x"], self.state[agent]["y"]])
            dist_to_target = float(np.linalg.norm(pos - self.target[agent]))

            # Progress Reward (×3)
            distance_delta = prev_dist - dist_to_target
            reward += distance_delta * 0.15

            # VMG Shaping (×3)
            leg_vmg_weight = 1.6 if self.state[agent]["current_leg"] == 1 else 1.25
            reward += max(vmg_norm, 0.0) * leg_vmg_weight * 6.0
            reward += min(vmg_norm, 0.0) * leg_vmg_weight * 3.0

            # Y-Axis Leg shaping (×3)
            leg_delta_y = float(self.state[agent]["y"] - prev_y)
            if self.state[agent]["current_leg"] == 1:
                reward += leg_delta_y * 0.15  # Su
            else:
                reward -= leg_delta_y * 0.15  # Giù

            # Costo Base (urgency ×3)
            reward -= 1.50

            # Penalità Timone (×3)
            reward -= (abs(rudder_input) ** 3) * 1.5
            rudder_delta = rudder_input - prev_rudder
            reward -= (abs(rudder_delta) ** 2) * 3.0

            # Vele: penalizza scossoni fuori assetto (×3)
            reward += (trim_eff - 0.72) * 6.0
            reward -= (abs(trim_delta) ** 1.5) * 4.5
            leg_trim_threshold = 0.18 if self.state[agent]["current_leg"] == 1 else 0.24
            if self.state[agent]["speed"] > 10.0 and trim_error > leg_trim_threshold:
                reward -= (trim_error - leg_trim_threshold) * 15.0

            # Trim vs VMG Synergy (×3)
            reward += max(vmg_norm, 0.0) * (trim_eff - 0.50) * 4.5

            # Dropped Foil Penalty (×3)
            if dropped_foil:
                reward -= 90.0
            if self.state[agent]["is_foiling"]:
                reward += (self.state[agent]["speed"] - self.foiling_drop_speed) * 0.6
            else:
                reward -= 0.9
            if self.state[agent]["speed"] < self.foiling_drop_speed:
                reward -= 1.5

            if distance_delta > 0:
                reward += self.state[agent]["speed"] * 0.15

            if dist_to_target < self.best_distance[agent]:
                self.best_distance[agent] = dist_to_target

            # -----------------------------------------------------------------
            # [6] Gate Check, Boundaries e Fine Regata
            # -----------------------------------------------------------------
            bx = self.state[agent]["x"]
            by = self.state[agent]["y"]

            # Hard Out of bounds (Dashed Lines / Edge of field)
            if (
                bx < self.boundaries["x_min"]
                or bx > self.boundaries["x_max"]
                or by < 0
                or by > self.field_length
            ):
                reward -= 600.0
                terminated = True
                self.state[agent]["termination_reason"] = "out_of_bounds"

            # Logic Gates e Prevenzione Loop (No-Backward-Sailing)
            gate_left = self.course_center_x - self.gate_width / 2.0
            gate_right = self.course_center_x + self.gate_width / 2.0

            if self.state[agent]["current_leg"] in [0, 1]:
                # Fasi di Bolina: la navigazione deve puntare verso l'alto (Y crescente)
                if by > self.state[agent]["max_y_reached"]:
                    self.state[agent]["max_y_reached"] = by
                
                # Se la barca perde oltre 150 metri rispetto alla massima quota Y raggiunta, 
                # sta navigando all'indietro o compiendo un loop mortale.
                if by < self.state[agent]["max_y_reached"] - 150.0:
                    reward -= 300.0
                    terminated = True
                    self.state[agent]["termination_reason"] = "anti_loop_penalty"

            elif self.state[agent]["current_leg"] == 2:
                # Fase di Poppa: la navigazione deve puntare verso il basso (Y decrescente)
                if by < self.state[agent]["min_y_reached"]:
                    self.state[agent]["min_y_reached"] = by

                if by > self.state[agent]["min_y_reached"] + 150.0:
                    reward -= 300.0
                    terminated = True
                    self.state[agent]["termination_reason"] = "anti_loop_penalty"

            # Leg 0: Start Gate
            if self.state[agent]["current_leg"] == 0:
                if by >= self.bottom_gate_y:
                    if gate_left <= bx <= gate_right:
                        self.state[agent]["current_leg"] = 1
                        reward += 300.0  # Partenza superata (×3)
                        # Target successivo: Boa di Bolina
                        self.target[agent] = np.array([self.round_marks[agent]["x"], self.top_gate_y])
                        pos = np.array([self.state[agent]["x"], self.state[agent]["y"]])
                        self.best_distance[agent] = float(np.linalg.norm(pos - self.target[agent]))
                    else:
                        reward -= 600.0
                        terminated = True
                        self.state[agent]["termination_reason"] = "missed_start_gate"

            elif self.state[agent]["current_leg"] == 1:
                if by >= self.top_gate_y and gate_left <= bx <= gate_right:
                    self.state[agent]["current_leg"] = 2
                    self.state[agent]["min_y_reached"] = by
                    reward += 900.0  # Boa Superata (×3)
                    self.state[agent]["post_round_pending"] = True

                    ext_offset = 60.0
                    down_offset = 40.0
                    if bx < self.course_center_x:
                        self.target[agent] = np.array(
                            [gate_left - ext_offset, self.top_gate_y - down_offset]
                        )
                    else:
                        self.target[agent] = np.array(
                            [gate_right + ext_offset, self.top_gate_y - down_offset]
                        )
                    self.previous_distance[agent] = float(
                        np.linalg.norm(np.array([bx, by]) - self.target[agent])
                    )
                    self.best_distance[agent] = self.previous_distance[agent]

            elif self.state[agent]["current_leg"] == 2:
                if self.state[agent].get("post_round_pending", False):
                    if dist_to_target <= max(self.target_radius * 1.2, 60.0):
                        self.state[agent]["post_round_pending"] = False
                        self.target[agent] = np.array(
                            [self.course_center_x, self.bottom_gate_y]
                        )
                        self.previous_distance[agent] = float(
                            np.linalg.norm(np.array([bx, by]) - self.target[agent])
                        )
                        self.best_distance[agent] = self.previous_distance[agent]

                if (
                    not self.state[agent].get("post_round_pending", False)
                    and by <= self.bottom_gate_y
                    and gate_left <= bx <= gate_right
                ):
                    efficiency = (
                        max(0, self.max_steps - self.step_count) / self.max_steps
                    )
                    reward += 3000.0 + efficiency * 1500.0
                    terminated = True
                    self.state[agent]["steps_to_target"] = self.step_count
                    self.state[agent]["termination_reason"] = "finished_race"

            if self.step_count >= self.max_steps:
                truncated = True
                self.state[agent]["termination_reason"] = "timeout"
                progress = 1.0 - (
                    self.best_distance[agent] / max(self.previous_distance[agent], 1.0)
                )
                if progress > 0:
                    reward += progress * 150.0

            # Conclusione e scrittura dizionari output base
            termination_reason = self.state[agent].get("termination_reason", None)
            finished_race = termination_reason == "finished_race"

            observations[agent] = self._get_obs(agent)
            rewards[agent] = reward
            terminations[agent] = terminated
            truncations[agent] = truncated
            infos[agent] = {
                "agent": agent,
                "distance_to_target": dist_to_target,
                "speed": self.state[agent]["speed"],
                "trim": self.state[agent]["sail_trim"],
                "trim_efficiency": trim_eff,
                "trim_target": trim_target,
                "trim_error": trim_error,
                "vmg": vmg,
                "vmg_norm": vmg_norm,
                "leg": self.state[agent]["current_leg"],
                "steps": self.step_count,
                "steps_to_target": self.state[agent].get("steps_to_target", None),
                "best_distance": self.best_distance[agent],
                "finished_race": finished_race,
                "termination_reason": termination_reason,
                "terminated": terminated,
                "truncated": truncated,
            }

        # ---------------------------------------------------------------------
        # RISOLUZIONE GLOBALE COLLISIONI 
        # (Post-calcolo distanze per simmetria fisica)
        # ---------------------------------------------------------------------
        collision_radius = 20.0
        agents_list = list(self.agents)
        for i in range(len(agents_list)):
            for j in range(i + 1, len(agents_list)):
                a1, a2 = agents_list[i], agents_list[j]
                if not (terminations.get(a1, True)) and not (terminations.get(a2, True)):
                    dx = self.state[a1]["x"] - self.state[a2]["x"]
                    dy = self.state[a1]["y"] - self.state[a2]["y"]
                    dist = float(np.linalg.norm([dx, dy]))
                    if dist < collision_radius:
                        terminations[a1], terminations[a2] = True, True
                        rewards[a1] -= 600.0
                        rewards[a2] -= 600.0
                        self.state[a1]["termination_reason"] = "collision"
                        self.state[a2]["termination_reason"] = "collision"
                        infos[a1]["terminated"] = True
                        infos[a2]["terminated"] = True
                        infos[a1]["termination_reason"] = "collision"
                        infos[a2]["termination_reason"] = "collision"

        self.agents = [
            a for a in self.agents if not (terminations[a] or truncations[a])
        ]
        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Richiama la classe di rendering specializzata matplotlib estratta in env_rendering.py."""
        if self.render_mode == "rgb_array":
            return self.renderer.render_frame()

    def close(self):
        """Chiude e dealloca le risorse del renderer visuale matplotlib collegato."""
        if self.renderer:
            self.renderer.close()
