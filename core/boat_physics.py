from __future__ import annotations
import numpy as np

from .sail_trim import (
    normalize_twa_deg,
    optimal_trim_for_twa,
    trim_efficiency,
    trim_speed_multiplier,
)

def compute_vmg_to_target(
    boat_x: float,
    boat_y: float,
    heading: float,
    speed: float,
    target_x: float,
    target_y: float,
) -> float:
    """Calcola la Velocity Made Good (VMG) verso il target corrente.

    Args:
        boat_x: Coordinata X della barca.
        boat_y: Coordinata Y della barca.
        heading: Prua della barca in radianti.
        speed: Velocità attuale in nodi.
        target_x: Coordinata X del bersaglio.
        target_y: Coordinata Y del bersaglio.

    Returns:
        Velocità in nodi proiettata lungo la direzione del bersaglio.
    """
    pos = np.array([boat_x, boat_y], dtype=np.float32)
    target_vec = np.array([target_x, target_y], dtype=np.float32) - pos
    
    target_norm = float(np.linalg.norm(target_vec))
    if target_norm < 1e-6:
        return 0.0

    target_unit = target_vec / target_norm
    boat_vel = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32) * speed
    return float(np.dot(boat_vel, target_unit))


def compute_polar_speed(
    apparent_wind_angle: float,
    wind_speed: float,
    is_foiling: bool,
    sail_trim: float,
    max_speed: float,
) -> tuple[float, float, float, float]:
    """Calcola la velocità teorica della barca basata sulla polare.

    Args:
        apparent_wind_angle: Angolo del vento apparente (usato qui come TWA).
        wind_speed: Velocità del vento locale in nodi.
        is_foiling: Stato attuale di volo (True se in foiling).
        sail_trim: Livello di regolazione delle vele [0, 1].
        max_speed: Velocità massima consentita.

    Returns:
        Una tupla contenente:
            - speed: Velocità calcolata in nodi.
            - trim_eff: Efficienza del trim attuale [0, 1].
            - optimal_trim: Trim ottimale per l'andatura corrente.
            - angle_deg: Angolo del vento in gradi (normalizzato).
    """
    # NOTE: AWA is TWA
    angle_deg = normalize_twa_deg(apparent_wind_angle)
        
    if is_foiling:
        if angle_deg < 45: speed_ratio = 0.0
        elif angle_deg < 55: speed_ratio = 1.0 + (angle_deg - 45) * 0.15
        elif angle_deg < 100: speed_ratio = 2.5 + (angle_deg - 55) * 0.024
        elif angle_deg < 140: speed_ratio = 3.6 + (angle_deg - 100) * 0.015
        elif angle_deg < 170: speed_ratio = 4.2 - (angle_deg - 140) * 0.03
        else: speed_ratio = 3.3 - (angle_deg - 170) * 0.25
    else:
        if angle_deg < 35: speed_ratio = 0.0
        elif angle_deg < 50: speed_ratio = 0.3 + (angle_deg - 35) * 0.03
        elif angle_deg < 110: speed_ratio = 0.75 + (angle_deg - 50) * 0.015
        elif angle_deg < 140: speed_ratio = 1.65 - (angle_deg - 110) * 0.01
        else: speed_ratio = 1.35 - (angle_deg - 140) * 0.005
        
    base_speed = min(speed_ratio * wind_speed, max_speed)
    optimal_trim = optimal_trim_for_twa(angle_deg, is_foiling)
    trim_eff = trim_efficiency(sail_trim, optimal_trim, is_foiling)
    speed = min(base_speed * trim_speed_multiplier(trim_eff, is_foiling), max_speed)
    
    return float(speed), float(trim_eff), float(optimal_trim), float(angle_deg)
