"""
wind_model.py
Campo di vento 2D con random walk temporale e spaziale.

- Vento "base" globale che evolve lentamente (random walk)
- Griglia NxN di perturbazioni locali che evolvono indipendentemente (mean-reverting)
- Interpolazione bilineare per ottenere il vento in qualsiasi punto (x, y)

Uso:
    wind = WindField(field_size=400)
    wind.reset(np_random, base_direction=1.2)

    # ogni step dell'ambiente:
    wind.step()
    direction, speed = wind.get_local_wind(x, y)
"""

import numpy as np


class WindField:
    """
    Campo di vento 2D con random walk.

    Parameters
    ----------
    field_size : float
        Dimensione del campo di gioco (metros). Default 400.
    grid_n : int
        Numero di celle per lato della griglia spaziale. Default 10.
    base_speed_range : tuple
        Intervallo (min, max) della velocità del vento base in nodi. Default (10, 18).
    temporal_drift_dir : float
        Massima variazione di direzione base per step (rad). Default 0.02.
    temporal_drift_speed : float
        Massima variazione di velocità base per step (kts). Default 0.3.
    spatial_std_dir : float
        Deviazione standard iniziale delle perturbazioni di direzione (rad). Default 0.15.
    spatial_std_speed : float
        Deviazione standard iniziale delle perturbazioni di velocità (kts). Default 2.0.
    spatial_corr : float
        Coefficiente di mean-reversion spaziale (0-1). Più alto = più persistenza. Default 0.95.
    """

    def __init__(
        self,
        field_size: float = 400,
        grid_n: int = 10,
        base_speed_range: tuple = (10, 18),
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

        # Stato interno (inizializzato in reset)
        self.base_direction: float = 0.0
        self.base_speed: float = float(np.mean(base_speed_range))
        self._delta_dir: np.ndarray = np.zeros((grid_n, grid_n))
        self._delta_speed: np.ndarray = np.zeros((grid_n, grid_n))
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def reset(self, np_random, base_direction: float | None = None) -> None:
        """
        Reimposta il campo di vento.

        Parameters
        ----------
        np_random : gymnasium.utils.seeding.np_random
            RNG condiviso con l'ambiente.
        base_direction : float | None
            Direzione base iniziale (rad). Se None, viene scelta casualmente.
        """
        self._rng = np_random

        self.base_speed = float(np_random.uniform(*self.base_speed_range))

        if base_direction is not None:
            self.base_direction = float(base_direction)
        else:
            self.base_direction = float(np_random.uniform(0, 2 * np.pi))

        # Perturbazioni spaziali iniziali (distribuzione normale)
        self._delta_dir = np_random.normal(
            0.0, self.spatial_std_dir, (self.grid_n, self.grid_n)
        ).astype(np.float32)
        self._delta_speed = np_random.normal(
            0.0, self.spatial_std_speed, (self.grid_n, self.grid_n)
        ).astype(np.float32)

    def step(self) -> None:
        """
        Avanza la simulazione del vento di un timestep.

        - Il vento base compie un random walk lento (direzione e velocità).
        - Le perturbazioni spaziali compiono un random walk mean-reverting verso 0.
        """
        rng = self._rng

        # --- Random walk del vento base ---
        self.base_direction += float(
            rng.uniform(-self.temporal_drift_dir, self.temporal_drift_dir)
        )
        self.base_direction = self.base_direction % (2 * np.pi)

        self.base_speed += float(
            rng.uniform(-self.temporal_drift_speed, self.temporal_drift_speed)
        )
        self.base_speed = float(
            np.clip(self.base_speed, self.base_speed_range[0], self.base_speed_range[1])
        )

        # --- Random walk mean-reverting delle perturbazioni spaziali ---
        noise_dir = rng.normal(0, self.spatial_std_dir * 0.1, (self.grid_n, self.grid_n))
        noise_speed = rng.normal(0, self.spatial_std_speed * 0.1, (self.grid_n, self.grid_n))

        self._delta_dir = (self.spatial_corr * self._delta_dir + noise_dir).astype(np.float32)
        self._delta_speed = (self.spatial_corr * self._delta_speed + noise_speed).astype(np.float32)

        # Clamp per evitare valori estremi
        self._delta_dir = np.clip(
            self._delta_dir, -self.spatial_std_dir * 2.5, self.spatial_std_dir * 2.5
        )
        self._delta_speed = np.clip(
            self._delta_speed, -self.spatial_std_speed * 2.5, self.spatial_std_speed * 2.5
        )

    def get_local_wind(self, x: float, y: float) -> tuple[float, float]:
        """
        Restituisce (direction, speed) del vento nella posizione (x, y)
        tramite interpolazione bilineare della griglia.

        Returns
        -------
        direction : float  [0, 2π)
        speed     : float  [3, 30] kts
        """
        n = self.grid_n

        # Posizione normalizzata nella griglia
        gx = float(np.clip(x / self.field_size * (n - 1), 0, n - 1 - 1e-9))
        gy = float(np.clip(y / self.field_size * (n - 1), 0, n - 1 - 1e-9))

        ix, iy = int(gx), int(gy)
        fx, fy = gx - ix, gy - iy

        # Interpolazione bilineare (direzione)
        local_delta_dir = (
            self._delta_dir[ix, iy]     * (1 - fx) * (1 - fy)
            + self._delta_dir[ix + 1, iy]   * fx       * (1 - fy)
            + self._delta_dir[ix, iy + 1]   * (1 - fx) * fy
            + self._delta_dir[ix + 1, iy + 1] * fx     * fy
        )

        # Interpolazione bilineare (velocità)
        local_delta_speed = (
            self._delta_speed[ix, iy]     * (1 - fx) * (1 - fy)
            + self._delta_speed[ix + 1, iy]   * fx       * (1 - fy)
            + self._delta_speed[ix, iy + 1]   * (1 - fx) * fy
            + self._delta_speed[ix + 1, iy + 1] * fx     * fy
        )

        direction = (self.base_direction + local_delta_dir) % (2 * np.pi)
        speed = float(np.clip(self.base_speed + local_delta_speed, 3.0, 30.0))

        return float(direction), speed

    def get_grid_arrows(self, n_arrows: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Utility per il rendering: restituisce coordinate e componenti delle frecce vento.

        Returns
        -------
        xs, ys : coordinate degli arrow centers (n_arrows x n_arrows)
        us, vs : componenti (u=est, v=nord) del vento
        """
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
