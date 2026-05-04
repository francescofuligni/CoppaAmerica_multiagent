# 🏆 CoppaAmerica Multiagent

Benvenuti nel progetto **CoppaAmerica Multiagent**, un simulatore 2D avanzato di regate veliche che utilizza tecniche di **Reinforcement Learning (PPO)** per addestrare imbarcazioni a competere in un ambiente dinamico e realistico.

---

## 📖 Descrizione del Programma

Il software simula una regata tra due imbarcazioni (agenti) che devono navigare all'interno di un campo di gara caratterizzato da vento variabile (modellato con un *random walk* 2D). L'obiettivo degli agenti è completare il percorso nel minor tempo possibile, ottimizzando la velocità e seguendo le regole tattiche della vela.

### ⚓ Come funziona la Regata

La competizione è strutturata in **3 fasi principali (Leg)** che mettono alla prova diverse capacità di navigazione:

1.  **Leg 1 (Bolina - Upwind)**: Le barche partono dalla zona inferiore del campo (Y ≈ 0-100m) e devono risalire il vento fino al **Top Gate** (Y = 2300m). Per completare la fase, è obbligatorio attraversare lo spazio compreso tra le due boe del cancello.
2.  **Leg 2 (Poppa - Downwind)**: Dopo aver girato la boa di bolina, le barche devono scendere velocemente verso il **Bottom Gate** (Y = 200m) con il vento in poppa.
3.  **Leg 3 (Giro Finale)**: Una volta attraversata la linea di arrivo, le imbarcazioni devono compiere una manovra di rotazione completa attorno alla boa finale per convalidare il successo della regata.

### ⛵ Competizione e Fisica

Le imbarcazioni non sono semplici icone, ma seguono un modello fisico complesso, basato su curve di performance veliche (VPP - Velocity Prediction Program):
*   **Controlli Continui**: Gli agenti controllano in tempo reale l'**angolo del timone** e il **trim delle vele**.
*   **Fisica delle Polari**: Le velocità target vengono calcolate in modo estremamente realistico interrogando curve polari, che definiscono le prestazioni ideali a ogni angolazione e intensità del vento.
*   **Foiling**: Le barche possono "volare" sull'acqua (foiling) raggiungendo velocità elevate. Manovre troppo brusche o angoli di vento errati possono causare la caduta dai foil (*drop foil*), con conseguente enorme perdita di velocità.
*   **Regole e Penalità**: Per incoraggiare un comportamento realistico, il sistema applica penalità severe per:
    *   **Collisioni**: Contatto fisico tra le imbarcazioni (tenendo conto del diritto di rotta sulle mure opposte).
    *   **Fuori Campo**: Uscita dai confini laterali o verticali del campo di gara.
    *   **Missed Gate**: Mancato passaggio all'interno dei cancelli obbligatori.
*   **Apprendimento**: Attraverso il *Self-Play*, le barche imparano non solo a navigare, ma anche a reagire alla posizione dell'avversario per ottenere vantaggi tattici.

---

## 📂 Struttura del Progetto

Il codice è profondamente modulare e separato in comparti logici ben definiti:

- `core/`: Contiene la simulazione fisica cruda. Qui si trova il modello del vento (`wind_model.py`), la dinamica della barca con le curve polari (`boat_physics.py`) e l'ottimizzazione aerodinamica (`sail_trim.py`).
- `env/`: Definisce l'ambiente Multi-Agent in stile Gymnasium. Il cuore del simulatore (`sailing_env.py`) gestisce cinematica, collisioni, e il calcolo dettagliato delle ricompense RL. Include i moduli grafici in `rendering.py`.
- `train_ppo.py` e `evaluate_ppo.py`: Contengono le implementazioni dirette per l'avvio degli allenamenti (tramite Stable-Baselines3) e la validazione del modello addestrato.
- `main.py`: Una comoda interfaccia a linea di comando (CLI) che orchestra l'avvio coordinato di training, test e rendering video in base ai parametri richiesti.
- `callbacks.py`: Monitoraggio personalizzato per i log in TensorBoard durante il training (statistiche su completamenti, velocità media, ecc.).

---

## 🚀 Istruzioni Comandi

Tutte le operazioni principali sono gestite tramite lo script `main.py`. Assicurati di aver attivato l'ambiente virtuale prima di procedere.

### 🛠️ Setup Iniziale
```bash
# Crea l'ambiente virtuale
python3 -m venv .venv

# Attiva l'ambiente
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Installa le dipendenze
pip install -r requirements.txt
```

### 🧠 Addestramento (Training)
Il sistema gestisce automaticamente il versionamento dei modelli in `models/`.
```bash
# Inizia un NUOVO addestramento da zero
python main.py --train-new

# Riprendi l'ultimo addestramento esistente
python main.py --train-resume

# Esempio con parametri personalizzati
python main.py --train-new --steps 500000 --n-envs 8
```

### 📺 Test e Visualizzazione
Genera video MP4 per osservare il comportamento degli agenti.
```bash
# Genera un singolo video della regata (usa l'ultimo modello salvato)
python main.py --video-file videos/demo.mp4

# Genera 5 video con differenti condizioni di vento (seed casuali)
python main.py --test-multi
```

### 📊 Monitoraggio
```bash
# Visualizza le curve di apprendimento su TensorBoard
python -m tensorboard.main --logdir ./sailing_tensorboard/
```

---
> **Nota**: I parametri di configurazione (velocità del vento, iperparametri RL, dimensioni del campo) sono modificabili centralmente nel file `config.yaml`.
