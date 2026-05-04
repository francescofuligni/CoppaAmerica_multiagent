"""Modello del campo di vento 2D con dinamiche temporali e spaziali."""

import numpy as np
from typing import Optional

class WindField:
    """Gestisce la simulazione stocastica del campo di vento tramite griglia spaziale."""

    def __init__(
        self,
        field_size: float = 400,
        grid_n: int = 10,
        base_speed_range: tuple = (15.0, 22.0),
        temporal_drift_dir: float = 0.02,
        temporal_drift_speed: float = 0.3,
        spatial_std_dir: float = 0.15,
        spatial_std_speed: float = 2.0,
        spatial_corr: float = 0.95,
    ):
        self.field_size = field_size
        self.grid_n = grid_n
        self.base_speed_range = base_speed_range
        self.temporal_drift_dir = temporal_drift_dir
        self.temporal_drift_speed = temporal_drift_speed
        self.spatial_std_dir = spatial_std_dir
        self.spatial_std_speed = spatial_std_speed
        self.spatial_corr = spatial_corr

        self.base_direction: float = 0.0
        self.base_speed: float = float(np.mean(base_speed_range))
        self._delta_dir: np.ndarray = np.zeros((grid_n, grid_n))
        self._delta_speed: np.ndarray = np.zeros((grid_n, grid_n))
        self._rng = np.random.default_rng()

    def reset(self, np_random, base_direction: Optional[float] = None) -> None:
        """Inizializza o reimposta il campo di vento."""
        self._rng = np_random

        self.base_speed = float(np_random.uniform(*self.base_speed_range))

        if base_direction is not None:
            self.base_direction = float(base_direction)
        else:
            self.base_direction = float(np_random.uniform(0, 2 * np.pi))

        self._delta_dir = np_random.normal(
            0.0, self.spatial_std_dir, (self.grid_n, self.grid_n)
        ).astype(np.float32)
        self._delta_speed = np_random.normal(
            0.0, self.spatial_std_speed, (self.grid_n, self.grid_n)
        ).astype(np.float32)

    def step(self) -> None:
        """Avanza la simulazione del vento di un passo temporale."""
        rng = self._rng

        # --- Random Walk del Vento Base ---
        # La direzione e velocità "di base" del vento variano nel tempo in modo continuo ma stocastico 
        # (Random Walk), creando i tipici "salti di vento" (shifts) che rendono la regata tattica.
        self.base_direction += float(
            rng.uniform(-self.temporal_drift_dir, self.temporal_drift_dir)
        )
        self.base_direction = self.base_direction % (2 * np.pi)

        self.base_speed += float(
            rng.uniform(-self.temporal_drift_speed, self.temporal_drift_speed)
        )
        self.base_speed = float(
            np.clip(self.base_speed, 15.0, self.base_speed_range[1])
        )

        # --- Raffiche Locali (Processo Mean-Reverting) ---
        # Per simulare raffiche di vento localizzate nello spazio, la griglia applica un rumore gaussiano 
        # e poi lo attenua nel tempo (spatial_corr). Questo fa sì che le raffiche nascano, vivano 
        # per qualche secondo in un'area, e poi si smorzino spontaneamente (Mean Reversion).
        noise_dir = rng.normal(0, self.spatial_std_dir * 0.1, (self.grid_n, self.grid_n))
        noise_speed = rng.normal(0, self.spatial_std_speed * 0.1, (self.grid_n, self.grid_n))

        self._delta_dir = (self.spatial_corr * self._delta_dir + noise_dir).astype(np.float32)
        self._delta_speed = (self.spatial_corr * self._delta_speed + noise_speed).astype(np.float32)

        self._delta_dir = np.clip(
            self._delta_dir, -self.spatial_std_dir * 2.5, self.spatial_std_dir * 2.5
        )
        self._delta_speed = np.clip(
            self._delta_speed, -self.spatial_std_speed * 2.5, self.spatial_std_speed * 2.5
        )

    def get_local_wind(self, x: float, y: float) -> tuple[float, float]:
        """Calcola la direzione e la velocità del vento locale tramite interpolazione bilineare."""
        n = self.grid_n

        gx = float(np.clip(x / self.field_size * (n - 1), 0, n - 1 - 1e-9))
        gy = float(np.clip(y / self.field_size * (n - 1), 0, n - 1 - 1e-9))

        ix, iy = int(gx), int(gy)
        fx, fy = gx - ix, gy - iy

        local_delta_dir = (
            self._delta_dir[ix, iy]     * (1 - fx) * (1 - fy)
            + self._delta_dir[ix + 1, iy]   * fx       * (1 - fy)
            + self._delta_dir[ix, iy + 1]   * (1 - fx) * fy
            + self._delta_dir[ix + 1, iy + 1] * fx     * fy
        )

        local_delta_speed = (
            self._delta_speed[ix, iy]     * (1 - fx) * (1 - fy)
            + self._delta_speed[ix + 1, iy]   * fx       * (1 - fy)
            + self._delta_speed[ix, iy + 1]   * (1 - fx) * fy
            + self._delta_speed[ix + 1, iy + 1] * fx     * fy
        )

        direction = (self.base_direction + local_delta_dir) % (2 * np.pi)
        speed = float(np.clip(self.base_speed + local_delta_speed, 15.0, 30.0))

        return float(direction), speed

    def get_grid_arrows(self, n_arrows: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Genera i vettori del vento per la visualizzazione grafica."""
        step = self.field_size / n_arrows
        xs, ys = np.meshgrid(
            np.arange(step / 2, self.field_size, step),
            np.arange(step / 2, self.field_size, step),
        )
        us = np.zeros_like(xs)
        vs = np.zeros_like(ys)
        for i in range(n_arrows):
            for j in range(n_arrows):
                d, s = self.get_local_wind(xs[i, j], ys[i, j])
                us[i, j] = s * np.cos(d)
                vs[i, j] = s * np.sin(d)
        return xs, ys, us, vs
