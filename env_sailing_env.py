"""
Modulo: env_sailing_env
=======================
Ambiente PettingZoo ParallelEnv con struttura modulare (core_* + renderer),
arricchito con la logica avanzata proveniente da sailing_env.py.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from core_boat_physics import compute_polar_speed, compute_vmg_to_target
from core_sail_trim import (
    action_to_trim_level,
    normalize_twa_deg,
    optimal_trim_for_twa,
    trim_efficiency,
    trim_level_to_action,
)
from core_wind_model import WindField
from env_rendering import SailingRenderer


class ImprovedSailingEnv(ParallelEnv):
    """Ambiente di regata multi-agent con dinamica e reward shaping avanzati."""

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

        # Geometria mappa rettangolare.
        self.field_length = float(phys_cfg.get("field_length", 4075.0))
        self.field_width = float(phys_cfg.get("field_width", 1482.0))
        self.field_diag = float(np.sqrt(self.field_width ** 2 + self.field_length ** 2))
        self.field_size = max(self.field_width, self.field_length)

        self.max_speed = 50.0
        self.max_wind = 30.0
        self.target_radius = 50.0
        self.dt = 1.0
        self.max_steps = 1800

        self.possible_agents = ["red_boat", "blue_boat"]
        self.agents = self.possible_agents[:]

        # Timone, trim e inerzia.
        self.max_turn_per_step = np.radians(25)
        self.min_turn_factor = float(phys_cfg.get("min_turn_factor", 0.12))
        self.max_rudder_delta_per_step = float(
            phys_cfg.get("max_rudder_delta_per_step", 0.25)
        )
        self.mark_round_margin = 25.0
        self.post_round_offset_x = 90.0
        self.post_round_offset_y = 120.0

        self.max_trim_delta_per_step = float(
            phys_cfg.get("max_trim_delta_per_step", 0.10)
        )
        self.default_trim_level = float(phys_cfg.get("default_trim_level", 0.60))

        self.displacement_inertia = float(
            phys_cfg.get("displacement_inertia", 0.85)
        )
        self.foiling_inertia = float(phys_cfg.get("foiling_inertia", 0.98))

        self.foiling_takeoff_speed = float(phys_cfg.get("foiling_takeoff_speed", 18.0))
        self.foiling_drop_speed = float(phys_cfg.get("foiling_drop_speed", 15.0))

        # Collisioni / hard rules / spin / rounding control.
        self.collision_radius = float(phys_cfg.get("collision_radius", 20.0))
        self.near_collision_radius = float(phys_cfg.get("near_collision_radius", 40.0))
        self.max_collision_correction = float(
            phys_cfg.get("max_collision_correction", 6.0)
        )
        self.collision_penalty_scale = float(
            phys_cfg.get("collision_penalty_scale", 20.0)
        )
        self.near_collision_penalty_scale = float(
            phys_cfg.get("near_collision_penalty_scale", 3.0)
        )
        self.port_tack_collision_multiplier = float(
            phys_cfg.get("port_tack_collision_multiplier", 1.6)
        )
        self.ttc_horizon = float(phys_cfg.get("ttc_horizon", 4.0))
        self.ttc_penalty_scale = float(phys_cfg.get("ttc_penalty_scale", 5.0))
        self.hard_violation_penalty = float(
            phys_cfg.get("hard_violation_penalty", 10_000.0)
        )
        self.hard_rules_warmup_steps = int(phys_cfg.get("hard_rules_warmup_steps", 20_000))
        self.boundary_hard_after_steps = int(phys_cfg.get("boundary_hard_after_steps", 15))

        self.spin_window_len = int(phys_cfg.get("spin_window_len", 30))
        self.spin_turn_threshold = np.radians(
            float(phys_cfg.get("spin_turn_threshold_deg", 450.0))
        )
        self.spin_min_progress = float(phys_cfg.get("spin_min_progress", 0.0))

        self.rounding_grace_steps = int(phys_cfg.get("rounding_grace_steps", 20))
        self.rounding_step_penalty_scale = float(
            phys_cfg.get("rounding_step_penalty_scale", 0.35)
        )
        self.rounding_step_penalty_cap = float(
            phys_cfg.get("rounding_step_penalty_cap", 20.0)
        )
        self.rounding_timeout_steps = int(phys_cfg.get("rounding_timeout_steps", 120))
        self.rounding_retry_penalty = float(phys_cfg.get("rounding_retry_penalty", 350.0))
        self.rounding_max_retries = int(phys_cfg.get("rounding_max_retries", 2))

        # Campo regata.
        self.course_center_x = self.field_width / 2.0
        self.boundaries = {
            "x_min": float(phys_cfg.get("x_min", 0.0)),
            "x_max": float(phys_cfg.get("x_max", self.field_width)),
        }
        self.top_gate_y = float(phys_cfg.get("top_gate_y", self.field_length - 200.0))
        self.bottom_gate_y = float(phys_cfg.get("bottom_gate_y", 200.0))
        self.gate_width = float(phys_cfg.get("gate_width", 300.0))

        # Osservazione arricchita a 20 feature (allineata al tuo env locale).
        self.observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for agent in self.possible_agents
        }

        self.state = {}
        self.target = {}
        self.wind_field = WindField(field_size=int(max(self.field_width, self.field_length)))
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
        return float(angle % (2 * np.pi))

    def _set_rounding_target(
        self,
        agent: str,
        gate_left: float,
        gate_right: float,
        gate_y: float,
        side: float,
        down_offset: float,
    ) -> None:
        ext_offset = 60.0
        if side < 0:
            self.target[agent] = np.array([gate_left - ext_offset, gate_y + down_offset])
        else:
            self.target[agent] = np.array([gate_right + ext_offset, gate_y + down_offset])

    def _start_rounding_segment(self, agent: str, segment: str, side: float) -> None:
        self.state[agent]["post_round_pending"] = True
        self.state[agent]["rounding_steps"] = 0
        self.state[agent]["rounding_retries"] = 0
        self.state[agent]["rounding_segment"] = segment
        self.state[agent]["rounding_side"] = float(side)

    def _apply_rounding_control(
        self,
        agent: str,
        reward: float,
        terminated: bool,
        hard_rules_active: bool,
        gate_left: float,
        gate_right: float,
    ):
        if terminated:
            return reward, terminated, 0.0

        if not self.state[agent].get("post_round_pending", False):
            self.state[agent]["rounding_steps"] = 0
            self.state[agent]["rounding_retries"] = 0
            self.state[agent]["rounding_segment"] = None
            return reward, terminated, 0.0

        self.state[agent]["rounding_steps"] = self.state[agent].get("rounding_steps", 0) + 1
        step_count = self.state[agent]["rounding_steps"]
        over_steps = max(0, step_count - self.rounding_grace_steps)
        round_pen = min(
            self.rounding_step_penalty_cap,
            over_steps * self.rounding_step_penalty_scale,
        )
        reward -= round_pen

        if step_count <= self.rounding_timeout_steps:
            return reward, terminated, round_pen

        self.state[agent]["rounding_retries"] = self.state[agent].get("rounding_retries", 0) + 1
        retries = self.state[agent]["rounding_retries"]
        reward -= self.rounding_retry_penalty * min(3, retries)

        if hard_rules_active and retries >= self.rounding_max_retries:
            reward -= self.hard_violation_penalty
            terminated = True
            self.state[agent]["termination_reason"] = "failed_rounding"
            return reward, terminated, round_pen

        self.state[agent]["rounding_steps"] = 0
        side = float(self.state[agent].get("rounding_side", 1.0))
        segment = self.state[agent].get("rounding_segment", None)
        if segment == "top_to_bottom":
            self._set_rounding_target(
                agent, gate_left, gate_right, self.top_gate_y, side, down_offset=-40.0
            )
        elif segment == "bottom_finish":
            self._set_rounding_target(
                agent, gate_left, gate_right, self.bottom_gate_y, side, down_offset=40.0
            )

        return reward, terminated, round_pen

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        elif not hasattr(self, "np_random"):
            self.np_random, _ = gym.utils.seeding.np_random()

        self.agents = self.possible_agents[:]
        self.step_count = 0

        if options and "wind_direction" in options:
            base_dir = float(options["wind_direction"])
        else:
            base_dir = 1.5 * np.pi
        self.wind_field.reset(self.np_random, base_direction=base_dir)

        for i, agent in enumerate(self.possible_agents):
            start_x = self.np_random.uniform(
                self.boundaries["x_min"] + 50, self.boundaries["x_max"] - 50
            )
            start_y = self.np_random.uniform(20.0, 100.0) + i * 20.0

            self.state[agent] = {
                "x": start_x,
                "y": start_y,
                "speed": 0.0,
                "heading": 0.0,
                "rudder_angle": 0.0,
                "sail_trim": self.default_trim_level,
                "is_foiling": False,
                "active_foil": 1.0,
                "dropped_foil_penalty_applied": False,
                "current_leg": 1,
                "post_round_pending": False,
                "spin_turn_window": [],
                "spin_progress_window": [],
                "boundary_outside_steps": 0,
                "rounding_steps": 0,
                "rounding_retries": 0,
                "rounding_segment": None,
                "rounding_side": 0,
            }

            rounding_side = float(self.np_random.choice([-1, 1]))
            round_mark_x = self.course_center_x + rounding_side * (self.gate_width / 2.0)
            self.round_marks[agent] = {"side": rounding_side, "x": round_mark_x}

            self.target[agent] = np.array([round_mark_x, self.top_gate_y])
            self.trajectory[agent] = [
                np.array([self.state[agent]["x"], self.state[agent]["y"]])
            ]
            pos = np.array([self.state[agent]["x"], self.state[agent]["y"]])
            self.previous_distance[agent] = float(np.linalg.norm(pos - self.target[agent]))
            self.best_distance[agent] = self.previous_distance[agent]

        for agent in self.possible_agents:
            tack_sign = float(self.np_random.choice([-1, 1]))
            start_heading = base_dir + np.pi + tack_sign * np.radians(50.0)
            start_heading += self.np_random.uniform(-np.radians(10), np.radians(10))
            self.state[agent]["heading"] = self._normalize_angle(start_heading)

            local_wind_dir, _ = self.wind_field.get_local_wind(
                self.state[agent]["x"], self.state[agent]["y"]
            )
            twa = (local_wind_dir + np.pi) - self.state[agent]["heading"]
            twa_deg = normalize_twa_deg(twa)
            self.state[agent]["sail_trim"] = optimal_trim_for_twa(twa_deg, False)

        observations = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def _get_obs(self, agent):
        pos = np.array([self.state[agent]["x"], self.state[agent]["y"]])
        dist_to_target = np.linalg.norm(pos - self.target[agent])
        bearing_to_target = np.arctan2(
            self.target[agent][1] - pos[1], self.target[agent][0] - pos[0]
        )

        local_wind_dir, local_wind_speed = self.wind_field.get_local_wind(
            self.state[agent]["x"], self.state[agent]["y"]
        )

        heading = self.state[agent]["heading"]
        rel_bearing = bearing_to_target - heading

        apparent_wind = (local_wind_dir + np.pi) - heading
        apparent_wind = self._normalize_angle(apparent_wind)
        if apparent_wind > np.pi:
            apparent_wind -= 2 * np.pi

        opponent = next((a for a in self.possible_agents if a != agent), None)
        if opponent is not None and opponent in self.state:
            opp_pos = np.array(
                [self.state[opponent]["x"], self.state[opponent]["y"]],
                dtype=np.float32,
            )
            rel_vec = np.array(
                [
                    (opp_pos[0] - pos[0]) / self.field_width,
                    (opp_pos[1] - pos[1]) / self.field_length,
                ],
                dtype=np.float32,
            )
            opp_dist_norm = float(np.linalg.norm(opp_pos - pos) / self.field_diag)
            speed_adv = float(
                (self.state[agent]["speed"] - self.state[opponent]["speed"])
                / self.max_speed
            )
            rel_opp_x = float(np.clip(rel_vec[0], -1.0, 1.0))
            rel_opp_y = float(np.clip(rel_vec[1], -1.0, 1.0))
            opp_dist_norm = float(np.clip(opp_dist_norm, 0.0, 1.0))
            speed_adv = float(np.clip(speed_adv, -1.0, 1.0))
        else:
            rel_opp_x = 0.0
            rel_opp_y = 0.0
            opp_dist_norm = 1.0
            speed_adv = 0.0

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
                dist_to_target / self.field_diag,
                np.sin(rel_bearing),
                np.cos(rel_bearing),
                float(self.state[agent]["rudder_angle"]),
                trim_level_to_action(self.state[agent]["sail_trim"]),
                1.0 if self.state[agent]["is_foiling"] else 0.0,
                self.state[agent]["active_foil"],
                1.0 if self.state[agent]["current_leg"] == 1 else -1.0,
                rel_opp_x,
                rel_opp_y,
                opp_dist_norm,
                speed_adv,
            ],
            dtype=np.float32,
        )

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
            if agent not in self.agents:
                continue

            pos = np.array([self.state[agent]["x"], self.state[agent]["y"]])
            prev_dist = np.linalg.norm(pos - self.target[agent])
            prev_y = float(self.state[agent]["y"])

            prev_rudder = self.state[agent].get("rudder_angle", 0.0)
            prev_trim = self.state[agent].get("sail_trim", self.default_trim_level)
            prev_heading = float(self.state[agent]["heading"])

            local_wind_dir, local_wind_speed = self.wind_field.get_local_wind(
                self.state[agent]["x"], self.state[agent]["y"]
            )

            rudder_raw = action[0] if hasattr(action, "__len__") else action
            trim_raw = (
                action[1] if (hasattr(action, "__len__") and len(action) > 1) else None
            )

            rudder_target = float(np.clip(rudder_raw, -1.0, 1.0))
            rudder_delta_limited = np.clip(
                rudder_target - prev_rudder,
                -self.max_rudder_delta_per_step,
                self.max_rudder_delta_per_step,
            )
            rudder_input = float(np.clip(prev_rudder + rudder_delta_limited, -1.0, 1.0))
            self.state[agent]["rudder_angle"] = rudder_input

            speed_factor = self.state[agent]["speed"] / self.max_speed
            effective_factor = self.min_turn_factor + (1.0 - self.min_turn_factor) * speed_factor
            turn_rate = rudder_input * effective_factor * self.max_turn_per_step
            self.state[agent]["heading"] = self._normalize_angle(
                self.state[agent]["heading"] + turn_rate * self.dt
            )

            heading_delta = self._normalize_angle(self.state[agent]["heading"] - prev_heading)
            if heading_delta > np.pi:
                heading_delta -= 2 * np.pi
            spin_turn_window = self.state[agent].setdefault("spin_turn_window", [])
            spin_turn_window.append(abs(float(heading_delta)))
            if len(spin_turn_window) > self.spin_window_len:
                spin_turn_window.pop(0)

            apparent_wind_angle = (local_wind_dir + np.pi) - self.state[agent]["heading"]
            twa_deg = normalize_twa_deg(apparent_wind_angle)

            if trim_raw is None:
                trim_target_level = optimal_trim_for_twa(
                    twa_deg, self.state[agent]["is_foiling"]
                )
            else:
                trim_target_level = action_to_trim_level(float(np.clip(trim_raw, -1.0, 1.0)))

            trim_delta = float(
                np.clip(
                    trim_target_level - prev_trim,
                    -self.max_trim_delta_per_step,
                    self.max_trim_delta_per_step,
                )
            )
            self.state[agent]["sail_trim"] = float(np.clip(prev_trim + trim_delta, 0.0, 1.0))

            if self.state[agent]["is_foiling"]:
                brake_factor = (abs(rudder_input) ** 1.5) * 0.05
                self.state[agent]["speed"] *= 1.0 - brake_factor
            else:
                brake_factor = (abs(rudder_input) ** 1.5) * 0.35
                self.state[agent]["speed"] *= 1.0 - brake_factor

            target_speed, trim_eff, trim_target, _ = compute_polar_speed(
                apparent_wind_angle=apparent_wind_angle,
                wind_speed=local_wind_speed,
                is_foiling=self.state[agent]["is_foiling"],
                sail_trim=self.state[agent]["sail_trim"],
                max_speed=self.max_speed,
            )

            current_inertia = (
                self.foiling_inertia if self.state[agent]["is_foiling"] else self.displacement_inertia
            )
            self.state[agent]["speed"] = (
                self.state[agent]["speed"] * current_inertia
                + target_speed * (1.0 - current_inertia)
            )

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
            trim_eff = trim_efficiency(
                self.state[agent]["sail_trim"],
                trim_target,
                self.state[agent]["is_foiling"],
            )
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
            displacement = self.state[agent]["speed"] * 0.514 * self.dt
            self.state[agent]["x"] += displacement * np.cos(self.state[agent]["heading"])
            self.state[agent]["y"] += displacement * np.sin(self.state[agent]["heading"])

            self.trajectory[agent].append(
                np.array([self.state[agent]["x"], self.state[agent]["y"]])
            )

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

                pos_curr = np.array([self.state[agent]["x"], self.state[agent]["y"]], dtype=np.float32)
                pos_other = np.array([self.state[other]["x"], self.state[other]["y"]], dtype=np.float32)
                delta = pos_curr - pos_other
                dist = float(np.linalg.norm(delta))

                if not (1e-6 < dist < near_collision_radius):
                    continue

                near_ratio = (near_collision_radius - dist) / near_collision_radius
                near_pen = near_ratio * self.near_collision_penalty_scale
                near_collision_penalty += near_pen
                rewards.setdefault(other, 0.0)
                rewards[other] -= near_pen
                if other in infos:
                    infos[other]["near_collision_penalty"] = infos[other].get(
                        "near_collision_penalty", 0.0
                    ) + near_pen

                delta_hat = delta / dist
                vel_curr = np.array(
                    [
                        np.cos(self.state[agent]["heading"]) * self.state[agent]["speed"],
                        np.sin(self.state[agent]["heading"]) * self.state[agent]["speed"],
                    ],
                    dtype=np.float32,
                )
                vel_other = np.array(
                    [
                        np.cos(self.state[other]["heading"]) * self.state[other]["speed"],
                        np.sin(self.state[other]["heading"]) * self.state[other]["speed"],
                    ],
                    dtype=np.float32,
                )
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
                            infos[other]["ttc_penalty"] = infos[other].get("ttc_penalty", 0.0) + ttc_pen
                            infos[other]["ttc_risk_count"] = infos[other].get("ttc_risk_count", 0) + 1

                if dist >= collision_radius:
                    continue

                overlap_ratio = (collision_radius - dist) / collision_radius
                base_penalty = overlap_ratio * self.collision_penalty_scale

                penalty_curr = base_penalty
                penalty_other = base_penalty

                opposite_tacks = (
                    self.state[agent]["active_foil"] * self.state[other]["active_foil"]
                ) < 0.0
                if opposite_tacks:
                    curr_is_port = self.state[agent]["active_foil"] < 0.0
                    other_is_port = self.state[other]["active_foil"] < 0.0
                    if curr_is_port and not other_is_port:
                        penalty_curr *= self.port_tack_collision_multiplier
                        row_violation_count += 1
                    elif other_is_port and not curr_is_port:
                        penalty_other *= self.port_tack_collision_multiplier
                        if other in infos:
                            infos[other]["row_violation_count"] = infos[other].get(
                                "row_violation_count", 0
                            ) + 1

                collision_penalty += penalty_curr
                collision_count += 1

                hard_collision = (
                    hard_rules_active
                    and dist < (collision_radius * 0.6)
                    and closing_speed > 6.0
                )
                if hard_collision:
                    hard_violation = True
                    hard_violation_reason = "collision"

                rewards.setdefault(other, 0.0)
                rewards[other] -= penalty_other
                if hard_collision:
                    rewards[other] -= self.hard_violation_penalty
                    terminations[other] = True
                    truncations[other] = False
                    if other in self.state:
                        self.state[other]["termination_reason"] = "collision"

                overlap = collision_radius - dist
                corr_mag = min(max_correction_per_boat, overlap * 0.5)
                direction = delta / dist
                correction = direction * corr_mag

                self.state[agent]["x"] += float(correction[0])
                self.state[agent]["y"] += float(correction[1])
                self.state[other]["x"] -= float(correction[0])
                self.state[other]["y"] -= float(correction[1])

                if other in observations:
                    observations[other] = self._get_obs(other)
                if other in infos:
                    infos[other]["collision_penalty"] = infos[other].get(
                        "collision_penalty", 0.0
                    ) + penalty_other
                    infos[other]["collision_count"] = infos[other].get("collision_count", 0) + 1
                    if hard_collision:
                        infos[other]["termination_reason"] = "collision"

            pos = np.array([self.state[agent]["x"], self.state[agent]["y"]])
            dist_to_target = np.linalg.norm(pos - self.target[agent])

            reward = 0.0
            terminated = False
            truncated = False
            self.state[agent]["termination_reason"] = None

            distance_delta = prev_dist - dist_to_target
            spin_progress_window = self.state[agent].setdefault("spin_progress_window", [])
            spin_progress_window.append(float(distance_delta))
            if len(spin_progress_window) > self.spin_window_len:
                spin_progress_window.pop(0)

            progress_gain = 1.8 if self.state[agent]["is_foiling"] else 0.65
            if distance_delta >= 0:
                reward += distance_delta * progress_gain
            else:
                reward += distance_delta * (progress_gain * 1.8)

            target_bearing = np.arctan2(
                self.target[agent][1] - pos[1], self.target[agent][0] - pos[0]
            )
            heading_error = self._normalize_angle(target_bearing - self.state[agent]["heading"])
            if heading_error > np.pi:
                heading_error -= 2 * np.pi
            heading_error_norm = abs(heading_error) / np.pi
            reward += (1.0 - heading_error_norm) * 1.2
            if heading_error_norm > 0.75:
                reward -= (heading_error_norm - 0.75) * 3.0

            leg_vmg_weight = 1.6 if self.state[agent]["current_leg"] == 1 else 1.25
            reward += max(vmg_norm, 0.0) * leg_vmg_weight * 8.5
            reward += min(vmg_norm, 0.0) * leg_vmg_weight * 4.0

            opponent = next((a for a in self.possible_agents if a != agent), None)
            if opponent is not None and opponent in self.state:
                opp_pos = np.array(
                    [self.state[opponent]["x"], self.state[opponent]["y"]], dtype=np.float32
                )
                opp_dist_to_target = float(np.linalg.norm(opp_pos - self.target[opponent]))
                tactical_advantage = (opp_dist_to_target - dist_to_target) / self.field_diag
                reward += float(np.clip(tactical_advantage, -1.0, 1.0)) * 1.5

            leg_delta_y = float(self.state[agent]["y"] - prev_y)
            if self.state[agent]["current_leg"] == 1:
                reward += leg_delta_y * 0.15
            else:
                reward -= leg_delta_y * 0.15

            reward -= 0.25
            reward -= collision_penalty
            reward -= near_collision_penalty
            reward -= ttc_penalty

            round_penalty = 0.0

            reward -= (abs(rudder_input) ** 3) * 2.0
            rudder_delta = rudder_input - prev_rudder
            reward -= (abs(rudder_delta) ** 2) * 5.0

            reward += (trim_eff - 0.72) * 8.5
            reward -= (abs(trim_delta) ** 1.5) * 5.5
            leg_trim_threshold = 0.18 if self.state[agent]["current_leg"] == 1 else 0.24
            if self.state[agent]["speed"] > 10.0 and trim_error > leg_trim_threshold:
                reward -= (trim_error - leg_trim_threshold) * 18.0

            reward += max(vmg_norm, 0.0) * (trim_eff - 0.50) * 5.0

            if dropped_foil:
                reward -= 40.0

            if self.state[agent]["is_foiling"]:
                speed_over_threshold = self.state[agent]["speed"] - self.foiling_drop_speed
                reward += speed_over_threshold * 1.1
            else:
                reward -= 1.5

            if self.state[agent]["speed"] < self.foiling_drop_speed:
                reward -= 3.0

            leg_target_speed = 20.0 if self.state[agent]["current_leg"] == 1 else 24.0
            speed_deficit = max(0.0, leg_target_speed - self.state[agent]["speed"])
            reward -= speed_deficit * 0.35

            if distance_delta > 0:
                reward += self.state[agent]["speed"] * 0.28

            if dist_to_target < self.best_distance[agent]:
                self.best_distance[agent] = dist_to_target

            bx = self.state[agent]["x"]
            by = self.state[agent]["y"]

            if bx < self.boundaries["x_min"] or bx > self.boundaries["x_max"]:
                self.state[agent]["boundary_outside_steps"] = self.state[agent].get(
                    "boundary_outside_steps", 0
                ) + 1
                reward -= 30.0
                if (
                    hard_rules_active
                    and self.state[agent]["boundary_outside_steps"] >= self.boundary_hard_after_steps
                ):
                    reward -= self.hard_violation_penalty
                    terminated = True
                    self.state[agent]["termination_reason"] = "boundary_violation"
            else:
                self.state[agent]["boundary_outside_steps"] = 0

            if bx < 0 or bx > self.field_width or by < 0 or by > self.field_length:
                reward -= self.hard_violation_penalty
                terminated = True
                self.state[agent]["termination_reason"] = "out_of_bounds"

            gate_left = self.course_center_x - self.gate_width / 2.0
            gate_right = self.course_center_x + self.gate_width / 2.0

            if self.state[agent]["current_leg"] == 1:
                crossed_top_line = prev_y < self.top_gate_y <= by
                if hard_rules_active and crossed_top_line and not (gate_left <= bx <= gate_right):
                    reward -= self.hard_violation_penalty
                    terminated = True
                    self.state[agent]["termination_reason"] = "missed_top_gate"

                if by >= self.top_gate_y and gate_left <= bx <= gate_right:
                    self.state[agent]["current_leg"] = 2
                    reward += 500.0
                    round_side = -1.0 if bx < self.course_center_x else 1.0
                    self._start_rounding_segment(agent, segment="top_to_bottom", side=round_side)
                    self._set_rounding_target(
                        agent,
                        gate_left,
                        gate_right,
                        self.top_gate_y,
                        round_side,
                        down_offset=-40.0,
                    )
                    self.previous_distance[agent] = np.linalg.norm(pos - self.target[agent])
                    self.best_distance[agent] = self.previous_distance[agent]

            elif self.state[agent]["current_leg"] == 2:
                if self.state[agent].get("post_round_pending", False):
                    if dist_to_target <= max(self.target_radius * 1.2, 60.0):
                        self.state[agent]["post_round_pending"] = False
                        self.state[agent]["rounding_segment"] = None
                        self.state[agent]["rounding_steps"] = 0
                        self.state[agent]["rounding_retries"] = 0
                        self.target[agent] = np.array([self.course_center_x, self.bottom_gate_y])
                        self.previous_distance[agent] = np.linalg.norm(pos - self.target[agent])
                        self.best_distance[agent] = self.previous_distance[agent]

                if (
                    (not self.state[agent].get("post_round_pending", False))
                    and by <= self.bottom_gate_y
                    and gate_left <= bx <= gate_right
                ):
                    self.state[agent]["current_leg"] = 3
                    reward += 500.0
                    round_side = -1.0 if bx < self.course_center_x else 1.0
                    self._start_rounding_segment(
                        agent, segment="bottom_finish", side=round_side
                    )
                    self._set_rounding_target(
                        agent,
                        gate_left,
                        gate_right,
                        self.bottom_gate_y,
                        round_side,
                        down_offset=40.0,
                    )
                    self.previous_distance[agent] = np.linalg.norm(pos - self.target[agent])
                    self.best_distance[agent] = self.previous_distance[agent]

            elif self.state[agent]["current_leg"] == 3:
                if self.state[agent].get("post_round_pending", False):
                    if dist_to_target <= max(self.target_radius * 1.2, 60.0):
                        efficiency = max(0, self.max_steps - self.step_count) / self.max_steps
                        reward += 2000.0 + efficiency * 1000.0
                        terminated = True
                        self.state[agent]["steps_to_target"] = self.step_count
                        self.state[agent]["termination_reason"] = "finished_race"
                        self.state[agent]["post_round_pending"] = False
                        self.state[agent]["rounding_segment"] = None
                        self.state[agent]["rounding_steps"] = 0
                        self.state[agent]["rounding_retries"] = 0

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
                self.state[agent]["termination_reason"] = "timeout"
                progress = 1.0 - (self.best_distance[agent] / max(self.previous_distance[agent], 1.0))
                if progress > 0:
                    reward += progress * 200.0

            if hard_rules_active and not terminated and self.state[agent]["current_leg"] != 3:
                turn_sum = float(np.sum(self.state[agent].get("spin_turn_window", [])))
                prog_sum = float(np.sum(self.state[agent].get("spin_progress_window", [])))
                if (
                    len(self.state[agent].get("spin_turn_window", [])) >= self.spin_window_len
                    and turn_sum >= self.spin_turn_threshold
                    and prog_sum <= self.spin_min_progress
                ):
                    reward -= self.hard_violation_penalty
                    terminated = True
                    self.state[agent]["termination_reason"] = "spin_violation"

            if hard_violation and not terminated:
                reward -= self.hard_violation_penalty
                terminated = True
                self.state[agent]["termination_reason"] = hard_violation_reason

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
                "collision_penalty": collision_penalty,
                "collision_count": collision_count,
                "near_collision_penalty": near_collision_penalty,
                "ttc_penalty": ttc_penalty,
                "ttc_risk_count": ttc_risk_count,
                "row_violation_count": row_violation_count,
                "rounding_penalty": round_penalty,
                "rounding_steps": self.state[agent].get("rounding_steps", 0),
                "rounding_retries": self.state[agent].get("rounding_retries", 0),
                "rounding_segment": self.state[agent].get("rounding_segment", None),
            }

            processed_agents.append(agent)

        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]
        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode == "rgb_array":
            return self.renderer.render_frame()

    def close(self):
        if self.renderer:
            self.renderer.close()
