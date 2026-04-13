"""
Modulo: core.sail_trim
======================
Fornisce un modello semplificato ma stabile per il trim delle vele in funzione
del True Wind Angle (TWA).

Convenzioni utilizzate:
- Il valore di trim è nel range [0, 1] (0 = vele molto lasche, 1 = vele molto cazzate).
- L'input fornito dalla policy (azione dell'agente) in [-1, 1] viene convertito nello spazio interno [0, 1].

Note Tecniche:
- In foiling, la sensibilità del trim è maggiore. Un errore di trim porta a una
  maggiore penalità in termini di efficienza, causando un decremento netto della velocità.
- In andatura "displacement" (non-foiling), l'effetto del trim è meno estremo.
"""

from __future__ import annotations
import numpy as np


def normalize_twa_deg(apparent_wind_angle: float) -> float:
    """
    Converte un angolo apparente espresso in radianti in gradi assoluti nel range [0, 180].

    Parameters:
        apparent_wind_angle (float): Angolo apparente in radianti (0 al centro della prua).

    Returns:
        float: L'angolo equivalente in gradi assoluti [0, 180].
    """
    diff = apparent_wind_angle % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return float(np.degrees(diff))


def optimal_trim_for_twa(twa_deg: float, is_foiling: bool) -> float:
    """
    Calcola il trim ideale [0, 1] in base all'angolo TWA attuale (in gradi).

    Parameters:
        twa_deg (float): True Wind Angle o Apparent Wind Angle normalizzato in gradi.
        is_foiling (bool): True se la barca sta attivamente volando sui foil.

    Returns:
        float: Il livello di trim ideale normalizzato tra 0 (lasco) e 1 (cazzato).

    Note:
        La curva di trim ideale differisce per le due andature (foiling vs displacement)
        poiché le barche AC75 necessitano di cazzare maggiormente le vele a parità di angolo reale.
    """
    points = (
        # (TWA deg, trim)
        [
            (45.0, 0.98),
            (65.0, 0.85),
            (90.0, 0.68),
            (130.0, 0.45),
            (160.0, 0.30),
            (180.0, 0.22),
        ]
        if is_foiling
        else [
            (35.0, 0.94),
            (60.0, 0.82),
            (90.0, 0.65),
            (130.0, 0.42),
            (160.0, 0.28),
            (180.0, 0.20),
        ]
    )

    angles = np.array([p[0] for p in points], dtype=np.float32)
    trims = np.array([p[1] for p in points], dtype=np.float32)
    return float(np.interp(np.clip(twa_deg, angles[0], angles[-1]), angles, trims))


def trim_efficiency(trim_value: float, optimal_trim: float, is_foiling: bool) -> float:
    """
    Misura l'efficienza del trim attuale basandosi su una distribuzione gaussiana
    attorno al valore ottimale.

    Parameters:
        trim_value (float): Livello di trim effettivo della barca [0, 1].
        optimal_trim (float): Livello di trim target per l'angolo corrente [0, 1].
        is_foiling (bool): True se la barca sta volando.

    Returns:
        float: Indice di efficienza (1.0 = perfetto, verso 0 = altamente imperfetto).

    Note:
        sigma=0.12 in foiling rende la barca molto più esigente sul trim preciso,
        sigma=0.16 in displacement perdona maggiori errori.
    """
    error = abs(float(trim_value) - float(optimal_trim))
    sigma = 0.12 if is_foiling else 0.16
    eff = np.exp(-0.5 * (error / sigma) ** 2)
    return float(np.clip(eff, 0.0, 1.0))


def trim_speed_multiplier(efficiency: float, is_foiling: bool) -> float:
    """
    Traduce l'efficienza del trim in un moltiplicatore di velocità per la polare.

    Parameters:
        efficiency (float): L'efficienza calcolata da trim_efficiency [0, 1].
        is_foiling (bool): True se la barca sta volando.

    Returns:
        float: Moltiplicatore di velocità da applicare alla velocità teorica massima.
    """
    if is_foiling:
        # Ponderazione estrema: senza un buon trim in foiling (0.60 base),
        # la barca poggia pesantemente o stalla.
        return float(0.60 + 0.65 * efficiency)

    # In displacement c'è maggiore resistenza fissa: l'effetto della vela imperfetta
    # influenza meno la dinamica lineare diretta (0.72 base).
    return float(0.72 + 0.42 * efficiency)


def action_to_trim_level(trim_action: float) -> float:
    """
    Mappa un'azione continua in arrivo dalla RL policy nello spazio trim effettivo della barca.

    Parameters:
        trim_action (float): L'output della policy RL nel range [-1, 1].

    Returns:
        float: Azione scalata nel range naturale del trim [0, 1].
    """
    return float(np.clip((trim_action + 1.0) * 0.5, 0.0, 1.0))


def trim_level_to_action(trim_level: float) -> float:
    """
    Esegue la mappatura inversa, utile ad esempio nell'osservazione dell'ambiente,
    per riportare il livello del trim interno [0, 1] nello spazio normalizzato per la rete neurale [-1, 1].

    Parameters:
        trim_level (float): Il trim interno nel range [0, 1].

    Returns:
        float: Azione scalata nel range [-1, 1].
    """
    return float(np.clip(trim_level * 2.0 - 1.0, -1.0, 1.0))
