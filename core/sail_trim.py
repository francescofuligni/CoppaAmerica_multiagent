"""Modello per il trim delle vele basato sull'angolo del vento reale (TWA)."""

from __future__ import annotations

import numpy as np


def normalize_twa_deg(apparent_wind_angle: float) -> float:
    """Converte l'angolo del vento in gradi assoluti [0, 180], indipendentemente dalle mura."""
    diff = apparent_wind_angle % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return float(np.degrees(diff))


def optimal_trim_for_twa(twa_deg: float, is_foiling: bool) -> float:
    """Determina il livello di trim ideale per un dato TWA tramite interpolazione sulle polari."""
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
    """Calcola l'efficienza della regolazione delle vele (gaussiana sul trim ottimale, più severa in foiling)."""
    error = abs(float(trim_value) - float(optimal_trim))
    sigma = 0.12 if is_foiling else 0.16
    eff = np.exp(-0.5 * (error / sigma) ** 2)
    return float(np.clip(eff, 0.0, 1.0))


def trim_speed_multiplier(efficiency: float, is_foiling: bool) -> float:
    """Converte l'efficienza del trim in un moltiplicatore di velocità."""
    if is_foiling:
        return float(0.60 + 0.65 * efficiency)
    return float(0.72 + 0.42 * efficiency)


def action_to_trim_level(trim_action: float) -> float:
    """Mappa l'azione della policy [-1, 1] nel livello di trim reale [0, 1]."""
    return float(np.clip((trim_action + 1.0) * 0.5, 0.0, 1.0))


def trim_level_to_action(trim_level: float) -> float:
    """Mappa un livello di trim reale [0, 1] nell'azione della policy [-1, 1]."""
    return float(np.clip(trim_level * 2.0 - 1.0, -1.0, 1.0))
