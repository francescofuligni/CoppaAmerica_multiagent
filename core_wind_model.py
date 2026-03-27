"""
Modulo: core.wind_model
=======================
Gestisce un campo di vento 2D con random walk spaziale e temporale.

Architettura:
- Vento base globale che compie un random walk temporale lento.
- Griglia (N x N) di perturbazioni spaziali interpolata per ogni posizione, 
  che evolve con approccio mean-reverting (ritorna fisiologicamente verso 0).
- Permette perturbazioni localizzate (salti di direzione o "raffiche" d'intensità).

Uso tipico nell'ambiente RL:
    wind = WindField(field_size=2500)
    wind.reset(np_random_generator, base_direction=1.5 * np.pi)
    ...
    # Durante step()
    wind.step()
    dir, speed = wind.get_local_wind(boat.x, boat.y)
"""

import numpy as np
from typing import Optional


class WindField:
    """
    Rappresenta il campo e la direzione del vento nell'ambiente di simulazione,
    modificando velocità e direzione nel tempo e nello spazio.
    """

    def __init__(
        self,
        field_size: float = 2500,
        grid_n: int = 10,
        base_speed_range: tuple = (15.0, 22.0),
        temporal_drift_dir: float = 0.02,
        temporal_drift_speed: float = 0.3,
        spatial_std_dir: float = 0.15,
        spatial_std_speed: float = 2.0,
        spatial_corr: float = 0.95,
    ):
        """
        Inizializza il campo di vento con le sue costanti di deriva temporale e spaziale.

        Parameters:
            field_size (float): Dimensione della regata (lato in metri). Default 2500.
            grid_n (int): Numero di nodi/celle per lato per l'interpolazione spaziale. Default 10.
            base_speed_range (tuple): Min/Max kts (velocità iniziale). Default (15.0, 22.0).
            temporal_drift_dir (float): Deviazione massima dell'angolo per step.
            temporal_drift_speed (float): Deviazione massima della velocità globale per step.
            spatial_std_dir (float): Std dev iniziale perturbazioni angolo.
            spatial_std_speed (float): Std dev iniziale perturbazioni velocità.
            spatial_corr (float): Persistenza mean-reverting della griglia (1.0 = infinito).
        """
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
        """
        Rigenera il layer di vento globale all'inizio di un nuovo episodio (regata).

        Parameters:
            np_random (Generator): Passa l'rng interno di Gymnasium per garantire 
                la riproducibilità tra episodi.
            base_direction (float|None): Se fornito, stabilizza la direzione media.
        """
        self._rng = np_random
        self.base_speed = float(np_random.uniform(*self.base_speed_range))

        if base_direction is not None:
            self.base_direction = float(base_direction)
        else:
            self.base_direction = float(np_random.uniform(0, 2 * np.pi))

        # Perturbazioni di rumore bianco alla griglia di sfondo (distribuzione normale)
        self._delta_dir = np_random.normal(
            0.0, self.spatial_std_dir, (self.grid_n, self.grid_n)
        ).astype(np.float32)
        self._delta_speed = np_random.normal(
            0.0, self.spatial_std_speed, (self.grid_n, self.grid_n)
        ).astype(np.float32)

    def step(self) -> None:
        """
        Avanza la simulazione del vento di uno step computazionale, aggiornando
        la base globale (random walk) e ricalcolando la degradazione delle perturbazioni 
        spaziali (mean-reverting a zero usando spatial_corr).
        """
        rng = self._rng

        # Deriva temporale del vento base
        self.base_direction += float(rng.uniform(-self.temporal_drift_dir, self.temporal_drift_dir))
        self.base_direction = self.base_direction % (2 * np.pi)

        self.base_speed += float(rng.uniform(-self.temporal_drift_speed, self.temporal_drift_speed))
        # Hard limits
        self.base_speed = float(np.clip(self.base_speed, 15.0, self.base_speed_range[1]))

        # Noise incrementale per griglia perturbazioni
        noise_dir = rng.normal(0, self.spatial_std_dir * 0.1, (self.grid_n, self.grid_n))
        noise_speed = rng.normal(0, self.spatial_std_speed * 0.1, (self.grid_n, self.grid_n))

        # Modello stocastico mean-reverting (AR1 process)
        self._delta_dir = (self.spatial_corr * self._delta_dir + noise_dir).astype(np.float32)
        self._delta_speed = (self.spatial_corr * self._delta_speed + noise_speed).astype(np.float32)

        # Clamping strutturale per non stravolgere la fisica della barca
        self._delta_dir = np.clip(
            self._delta_dir, -self.spatial_std_dir * 2.5, self.spatial_std_dir * 2.5
        )
        self._delta_speed = np.clip(
            self._delta_speed, -self.spatial_std_speed * 2.5, self.spatial_std_speed * 2.5
        )

    def get_local_wind(self, x: float, y: float) -> tuple[float, float]:
        """
        Estrapola i parametri del vento locali data una posizione XY usando 
        interpolazione bilineare tra i punti più vicini nella griglia di noise.

        Parameters:
            x, y (float): Coordinate metriche della barca nell'ambiente.

        Returns:
            tuple (direction, speed):
                direction -> float rad [0, 2pi)
                speed     -> float nodi (kts)
        """
        n = self.grid_n

        gx = float(np.clip(x / self.field_size * (n - 1), 0, n - 1 - 1e-9))
        gy = float(np.clip(y / self.field_size * (n - 1), 0, n - 1 - 1e-9))

        ix, iy = int(gx), int(gy)
        fx, fy = gx - ix, gy - iy

        # Interpolazione bilineare offset direzione
        local_delta_dir = (
            self._delta_dir[ix, iy]           * (1 - fx) * (1 - fy)
            + self._delta_dir[ix + 1, iy]     * fx       * (1 - fy)
            + self._delta_dir[ix, iy + 1]     * (1 - fx) * fy
            + self._delta_dir[ix + 1, iy + 1] * fx       * fy
        )

        # Interpolazione bilineare offset velocità
        local_delta_speed = (
            self._delta_speed[ix, iy]           * (1 - fx) * (1 - fy)
            + self._delta_speed[ix + 1, iy]     * fx       * (1 - fy)
            + self._delta_speed[ix, iy + 1]     * (1 - fx) * fy
            + self._delta_speed[ix + 1, iy + 1] * fx       * fy
        )

        direction = (self.base_direction + local_delta_dir) % (2 * np.pi)
        speed = float(np.clip(self.base_speed + local_delta_speed, 15.0, 30.0))

        return float(direction), speed

    def get_grid_arrows(self, n_arrows: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Utility per il rendering grafico di Matplotlib. 
        Calcola i vettori su griglia uniforme da rappresentare graficamente.

        Parameters:
            n_arrows (int): Risoluzione del sampling della griglia. Default 8x8.

        Returns:
            tuple (xs, ys, us, vs): Array di meshgrid (pos coordinate) e (uv vector dims).
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
