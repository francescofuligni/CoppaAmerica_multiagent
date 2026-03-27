"""
Modulo: core_boat_physics
=========================
Contiene le funzioni pure e deterministiche che calcolano il comportamento 
aerodinamico, le velocità polari e i vettori di movimento barca (es. VMG).

Separando queste logiche matematiche e trigonometriche dall'ambiente (Gymnasium), 
il file `env_sailing_env.py` rimane focalizzato esclusivamente sulla gestione dello 
stato, lo step incrementale e le reward.

Note Tecniche:
- Foiling vs Displacement: I modelli AC75 presentano uno scalino drammatico
  nelle prestazioni "foiling" e "displacement" (in acqua). Le curve di polare
  sono state mappate usando rapporti espliciti (speed_ratio) per modellare:
  1. Il divieto fisico di puntare troppo a monte del vento (stall limit).
  2. L'esplosione di velocità dai 50 ai 100 gradi.
  3. Il calo di portanza a fil di ruota (180 gradi), per cui navigare in poppa 
     esatta è contro-producente, forzando la barca a procedere "a zig zag"
     in VMG (Velocity Made Good) giù per il vento.
"""

from __future__ import annotations
import numpy as np

# Moduli core dipendenti
from core_sail_trim import normalize_twa_deg, optimal_trim_for_twa, trim_efficiency, trim_speed_multiplier


def compute_vmg_to_target(
    boat_x: float, 
    boat_y: float, 
    heading: float, 
    speed: float, 
    target_x: float, 
    target_y: float
) -> float:
    """
    Calcola il VMG (Velocity Made Good), ovvero la componente della velocità 
    effettivamente puntata in direzione del target spaziale fornito.

    Questo valore guida la "vera" competizione velica: non conta solo andare 
    veloci, ma quanto di quella velocità si tramuti in un avvicinamento
    attivo al traguardo corrente (es. una boa).

    Parameters:
        boat_x (float): Coordinata X della barca.
        boat_y (float): Coordinata Y della barca.
        heading (float): Angolo di prua in radianti.
        speed (float): Velocità attuale in nodi.
        target_x (float): Coordinata X del target (es. boa).
        target_y (float): Coordinata Y del target (es. boa).

    Returns:
        float: VMG in nodi. Può essere negativo se la barca si allontana dal target.
    """
    pos = np.array([boat_x, boat_y], dtype=np.float32)
    target_vec = np.array([target_x, target_y], dtype=np.float32) - pos
    
    target_norm = float(np.linalg.norm(target_vec))
    if target_norm < 1e-6:
        # Se siamo perfettamente sopra il bersaglio, l'avvicinamento aggiuntivo è nullo
        return 0.0

    target_unit = target_vec / target_norm
    boat_vel = np.array([np.cos(heading), np.sin(heading)], dtype=np.float32) * speed
    
    # Proiezione scalare (Dot product) del vettore velocità sul vettore direzionale
    return float(np.dot(boat_vel, target_unit))


def compute_polar_speed(
    apparent_wind_angle: float,
    wind_speed: float,
    is_foiling: bool,
    sail_trim: float,
    max_speed: float
) -> tuple[float, float, float, float]:
    """
    Estrapola la velocità teorica sviluppata e i dati di efficienza
    valutando l'angolo del vento apparente (usato storicamente come TWA proxy qui),
    la modalità foiling, e il posizionamento delle vele (trim).

    Parameters:
        apparent_wind_angle (float): Angolo di incidenza del vento sulla prua in radianti.
        wind_speed (float): Intensità del vento misurata alla posizione in nodi.
        is_foiling (bool): Determina la curva polare da applicare.
        sail_trim (float): Livello misurato di lasco/cazzato [0, 1].
        max_speed (float): Hard cap limit alla massima velocità impostabile.

    Returns:
        tuple[float, float, float, float]:
            - speed (float): Nuova velocità target polare.
            - trim_eff (float): Efficienza corrente della vela [0, 1].
            - optimal_trim (float): Target ottimale di vela suggerito [0, 1].
            - angle_deg (float): True Wind Angle (modulo 180 gradi) decodificato in gradi.
            
    Note Tecniche:
        La curva per barche AC75 prevede uno yield ratio (multiplo della velocità del vento)
        estremamente severo:
          - A < 45 gradi vi è stallo netto (ratio=0).
          - A 100 gradi si esprimono le performance maggiori (> 3.5x velocità del vento).
    """
    # NOTA: apparent_wind_angle è in realtà mappabile come True Wind Angle (TWA) 
    # nel layer termodinamico di questo ambiente. Normalizziamo in gradi [0, 180].
    angle_deg = normalize_twa_deg(apparent_wind_angle)
        
    if is_foiling:
        # Foiling (AC75-like): Impossibile stringere il vento puro, si vola solo dai 45-50° in sù
        if angle_deg < 45:
            speed_ratio = 0.0
        elif angle_deg < 55:
            speed_ratio = 1.0 + (angle_deg - 45) * 0.15
        elif angle_deg < 100:
            speed_ratio = 2.5 + (angle_deg - 55) * 0.024
        elif angle_deg < 140:
            speed_ratio = 3.6 + (angle_deg - 100) * 0.015
        elif angle_deg < 170:
            speed_ratio = 4.2 - (angle_deg - 140) * 0.03 # Raggiungono un picco ma calano dopo i 140
        else:
            speed_ratio = 3.3 - (angle_deg - 170) * 0.25 # Stalla se va completamente in poppa filata per l'ombra
    else:
        # Displacement: rotta in acqua, l'andatura di poppa funziona ma è molto più lenta
        if angle_deg < 35:
            speed_ratio = 0.0
        elif angle_deg < 50:
            speed_ratio = 0.3 + (angle_deg - 35) * 0.03
        elif angle_deg < 110:
            speed_ratio = 0.75 + (angle_deg - 50) * 0.015
        elif angle_deg < 140:
            speed_ratio = 1.65 - (angle_deg - 110) * 0.01
        else:
            speed_ratio = 1.35 - (angle_deg - 140) * 0.005 # A 180 gradi viaggia tranquilla (non inziale ombra)
        
    base_speed = min(speed_ratio * wind_speed, max_speed)
    
    # Processo aerodinamico: riduzione per inefficienza
    optimal_trim = optimal_trim_for_twa(angle_deg, is_foiling)
    trim_eff = trim_efficiency(sail_trim, optimal_trim, is_foiling)
    
    speed = min(base_speed * trim_speed_multiplier(trim_eff, is_foiling), max_speed)
    
    return float(speed), float(trim_eff), float(optimal_trim), float(angle_deg)
