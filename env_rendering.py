"""
Modulo: env_rendering
======================
Separazione delle responsabilità visive per matplotlib.
Estrae la logica grafica da `env_sailing_env.py` per alleggerire il codice dell'ambiente.
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class SailingRenderer:
    """
    Gestisce tutto il rendering visivo tramite Matplotlib per restituire
    un rgb_array per i layer di video creation.
    """

    def __init__(self, env):
        """
        Inizializza il renderer interfacciandolo con un'istanza dell'ambiente.

        Parameters:
            env (ImprovedSailingEnv): Referenza diretta all'environment 
                per poterne leggere costanti, stati e traiettorie in tempo reale.
        """
        self.env = env
        self.fig = None
        self.ax = None

    def render_frame(self) -> np.ndarray:
        """
        Disegna lo scenario corrente (confini, gates, vento, barche e telemetria)
        e lo rasterizza in un array numpy RGB.

        Returns:
            np.ndarray: L'immagine del frame a forma (H, W, 3).
        """
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.fig.clf()  # Clear memory ad ogni frame
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(0, self.env.field_size)
        self.ax.set_ylim(0, self.env.field_size)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#a0d8ef')

        # --- Frecce vento di base sul campo ---
        xs, ys, us, vs = self.env.wind_field.get_grid_arrows(n_arrows=8)
        speeds = np.sqrt(us**2 + vs**2)
        self.ax.quiver(
            xs, ys, us, vs, speeds, cmap='Blues', alpha=0.55,
            scale=220, width=0.003, headwidth=4, headlength=5
        )

        colors_map = {'red_boat': 'red', 'blue_boat': 'blue'}

        # --- Disegna Boundaries Esterne (Confini penalità) ---
        self.ax.plot(
            [self.env.boundaries['x_min'], self.env.boundaries['x_min']], 
            [0, self.env.field_size], 'r--', linewidth=2, alpha=0.5, label='Boundary'
        )
        self.ax.plot(
            [self.env.boundaries['x_max'], self.env.boundaries['x_max']], 
            [0, self.env.field_size], 'r--', linewidth=2, alpha=0.5
        )

        # --- Disegna Gates (Cancelli virtuali e Boe fisiche) ---
        gate_left = self.env.course_center_x - self.env.gate_width / 2.0
        gate_right = self.env.course_center_x + self.env.gate_width / 2.0

        # Top Gate (Bolina)
        self.ax.plot([gate_left, gate_right], [self.env.top_gate_y, self.env.top_gate_y], 'g--', alpha=0.3)
        self.ax.plot(gate_left, self.env.top_gate_y, 'go', markersize=8, label='Gate Mark')
        self.ax.plot(gate_right, self.env.top_gate_y, 'go', markersize=8)

        # Bottom Gate (Poppa / Arrivo)
        self.ax.plot([gate_left, gate_right], [self.env.bottom_gate_y, self.env.bottom_gate_y], 'g--', alpha=0.3)
        self.ax.plot(gate_left, self.env.bottom_gate_y, 'bo', markersize=8)
        self.ax.plot(gate_right, self.env.bottom_gate_y, 'bo', markersize=8)

        info_lines = []
        
        # --- Disegna Imbarcazioni ---
        for agent in self.env.possible_agents:
            agent_color = colors_map.get(agent, 'black')

            # Storico traiettoria
            if agent in self.env.trajectory and len(self.env.trajectory[agent]) > 1:
                traj = np.array(self.env.trajectory[agent])
                self.ax.plot(traj[:, 0], traj[:, 1], '-', color=agent_color, alpha=0.5, linewidth=2)

            # Draw barca
            if agent in self.env.state:
                bx, by = self.env.state[agent]['x'], self.env.state[agent]['y']
                hdg = self.env.state[agent]['heading']

                boat_size = 15
                boat_points = np.array([[boat_size, 0], [-boat_size / 2, boat_size / 2], [-boat_size / 2, -boat_size / 2]])
                rot = np.array([[np.cos(hdg), -np.sin(hdg)],
                                [np.sin(hdg),  np.cos(hdg)]])
                boat_points = boat_points @ rot.T + np.array([bx, by])

                # Visual: se in foiling il contorno/corpo diventa cyan
                is_foiling = self.env.state[agent].get('is_foiling', False)
                boat_color = 'cyan' if is_foiling else agent_color
                boat_edge = agent_color if is_foiling else 'darkgray'
                boat = patches.Polygon(boat_points, closed=True, facecolor=boat_color, edgecolor=boat_edge, linewidth=2)
                self.ax.add_patch(boat)

                # Active foil text indication (telemetria visiva sopra la barca)
                foil_side = "Port" if self.env.state[agent].get('active_foil', 1.0) == 1.0 else "Stbd"
                foil_str = f"FOILING ({foil_side})" if is_foiling else f"HULL ({foil_side})"
                self.ax.text(bx, by - 25, foil_str, fontsize=8, color='magenta', fontweight='bold', ha='center')

                # Distanze, passi
                dist = np.linalg.norm(np.array([bx, by]) - self.env.target[agent])
                steps = self.env.state[agent].get('steps_to_target', self.env.step_count)
                info_lines.append(f"{agent}: {dist:.0f}m ({steps}stp)")

                # Timone
                rudder_input = self.env.state[agent].get('rudder_angle', 0.0)
                rudder_deg = int(rudder_input * 25)
                rudder_color = 'red' if abs(rudder_input) > 0.5 else 'black'
                self.ax.text(bx + 18, by + 18, f"{rudder_deg:+d}°", fontsize=7, color=rudder_color, fontweight='bold')

                # Trim Vele
                trim_percent = int(self.env.state[agent].get('sail_trim', self.env.default_trim_level) * 100)
                self.ax.text(bx + 18, by + 7, f"Trim:{trim_percent:02d}%", fontsize=7, color='navy', fontweight='bold')

        # --- Box Informativi UI Superiore ---
        ref_agent = self.env.possible_agents[0]
        if ref_agent in self.env.state:
            local_wd, local_ws = self.env.wind_field.get_local_wind(
                self.env.state[ref_agent]['x'], self.env.state[ref_agent]['y']
            )
            wind_deg = (90 - np.degrees(local_wd)) % 360
            dist_to_gate = abs(self.env.target[ref_agent][1] - self.env.state[ref_agent]['y'])

            winner_text = ""
            finished_agents = {a: self.env.state[a]['steps_to_target'] for a in self.env.possible_agents if 'steps_to_target' in self.env.state[a]}
            if finished_agents:
                winner_agent = min(finished_agents, key=finished_agents.get)
                winner_text = f"WIN: {winner_agent} | "

            full_title = winner_text + " | ".join(info_lines)

            # Titolo principale (telemetria HUD)
            self.ax.set_title(
                f"{full_title}\n"
                f"Leg: {self.env.state[ref_agent].get('current_leg', 1)}/2 | "
                f"Speed: {self.env.state[ref_agent]['speed']:.1f} kts | "
                f"Trim: {self.env.state[ref_agent].get('sail_trim', self.env.default_trim_level) * 100:.0f}% | "
                f"Dist Y to Gate: {dist_to_gate:.0f}m",
                fontsize=10, weight='bold'
            )

            # Box Info Vento Numeriche
            wind_text = (
                f"Wind (base)\n"
                f"Dir: {(90 - np.degrees(self.env.wind_field.base_direction)) % 360:.0f}\u00b0\n"
                f"Speed: {self.env.wind_field.base_speed:.1f} kts\n"
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

            # Rosa Dei Venti Inset Grafico
            inset_ax = self.fig.add_axes([0.78, 0.78, 0.16, 0.16], polar=True)
            inset_ax.set_theta_zero_location('N')
            inset_ax.set_theta_direction(-1)
            inset_ax.set_rticks([])
            inset_ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
            inset_ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=6)
            inset_ax.set_title('Wind', fontsize=7, pad=2)

            compass_base = np.pi / 2 - self.env.wind_field.base_direction
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
        
        # Converte il buffer grafico in un array numpy RGB per esportazione video (esito compatibile cross-platform)
        try:
            image = np.array(self.fig.canvas.buffer_rgba(), copy=True)[:, :, :3]
        except AttributeError:
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = np.array(image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,)), copy=True)

        return image

    def close(self):
        """Libera attivamente la memoria di matplotib allocata."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
