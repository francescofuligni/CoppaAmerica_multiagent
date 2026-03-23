"""
sail_trim.py
Modello semplice ma stabile per il trim vele in funzione del True Wind Angle (TWA).

Convenzioni:
- trim in [0, 1] (0 = vele molto lasche, 1 = vele molto cazzate)
- input policy in [-1, 1] viene convertito in [0, 1]
"""

from __future__ import annotations

import numpy as np


def normalize_twa_deg(apparent_wind_angle: float) -> float:
    """Converte l'angolo in gradi assoluti [0, 180]."""
    diff = apparent_wind_angle % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return float(np.degrees(diff))


def optimal_trim_for_twa(twa_deg: float, is_foiling: bool) -> float:
    """Restituisce il trim ideale in [0, 1] per il TWA dato."""
    points = (
        # (TWA deg, trim)
        [(45.0, 0.98), (65.0, 0.85), (90.0, 0.68), (130.0, 0.45), (160.0, 0.30), (180.0, 0.22)]
        if is_foiling
        else [(35.0, 0.94), (60.0, 0.82), (90.0, 0.65), (130.0, 0.42), (160.0, 0.28), (180.0, 0.20)]
    )

    angles = np.array([p[0] for p in points], dtype=np.float32)
    trims = np.array([p[1] for p in points], dtype=np.float32)
    return float(np.interp(np.clip(twa_deg, angles[0], angles[-1]), angles, trims))


def trim_efficiency(trim_value: float, optimal_trim: float, is_foiling: bool) -> float:
    """
    Misura efficienza del trim in [0, 1].
    In foiling la sensibilita' all'errore e' maggiore.
    """
    error = abs(float(trim_value) - float(optimal_trim))
    sigma = 0.12 if is_foiling else 0.16
    # Curva gaussiana: 1.0 in centro, decrescita dolce lontano dall'ottimo.
    eff = np.exp(-0.5 * (error / sigma) ** 2)
    return float(np.clip(eff, 0.0, 1.0))


def trim_speed_multiplier(efficiency: float, is_foiling: bool) -> float:
    """Converte l'efficienza trim in moltiplicatore velocita'."""
    if is_foiling:
        # Foiling: premio forte all'allineamento perfetto, decadimento netto se fuori trim.
        return float(0.60 + 0.65 * efficiency)
    # Displacement: effetto leggermente meno estremo.
    return float(0.72 + 0.42 * efficiency)


def action_to_trim_level(trim_action: float) -> float:
    """Mappa azione policy [-1, 1] in livello trim [0, 1]."""
    return float(np.clip((trim_action + 1.0) * 0.5, 0.0, 1.0))


def trim_level_to_action(trim_level: float) -> float:
    """Mappa livello trim [0, 1] in azione policy [-1, 1]."""
    return float(np.clip(trim_level * 2.0 - 1.0, -1.0, 1.0))
