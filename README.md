# CoppaAmerica Multiagent

Simulatore 2D di regata velica con Reinforcement Learning (PPO).  
Un agente (o più, in sviluppo) impara a navigare verso un obiettivo in un campo con vento variabile, seguendo le regole della Coppa America.

---

## Requisiti

- Python **3.12+**
- Le dipendenze sono elencate in `requirements.txt`

---

## Setup (prima volta)

```bash
# 1. Clona il repo
git clone https://github.com/francescofuligni/CoppaAmerica_multiagent.git
cd CoppaAmerica_multiagent

# 2. Crea l'ambiente virtuale
python3 -m venv .venv

# 3. Attiva l'ambiente virtuale
source .venv/bin/activate          # Linux / macOS
# oppure:
.venv\Scripts\activate             # Windows

# 4. Installa le dipendenze
pip install -r requirements.txt
```

> **Nota:** non caricare mai la cartella `.venv/` su GitHub (già in `.gitignore`).

---

## Variabili d'ambiente

Crea un file `.env` nella root del progetto (non viene tracciato da git):

```ini
TOTAL_TIMESTEPS=500000
N_ENVS=4
MODEL_PATH=models/sailing_ppo_improved
VIDEO_FILE=videos/sailing_demo.mp4
TENSORBOARD_LOG=./sailing_tensorboard/
```

Un template è già incluso come `.env.example` (quando lo creerete).

---

## Avvio

Assicurati di avere l'ambiente virtuale attivo (`source .venv/bin/activate`).

### Training + video (default)
```bash
.venv/bin/python main.py
```

### Solo training (forza il ricalcolo anche se il modello esiste già)
```bash
.venv/bin/python main.py --train --steps 500000 --n-envs 4
```

### Solo video (usa un modello già salvato)
```bash
.venv/bin/python main.py --model-path models/sailing_ppo_improved --video-file videos/sailing_demo.mp4
```

### Visualizzare le curve di training con TensorBoard
```bash
.venv/bin/python -m tensorboard.main --logdir ./sailing_tensorboard/
```

---

## Struttura del progetto

```
CoppaAmerica_multiagent/
│
├── main.py               # Entry point CLI: avvia training e/o generazione video
├── train_ppo.py          # Training PPO con ambienti paralleli (SuperSuit + SB3)
├── evaluate_ppo.py       # Carica il modello e genera il video MP4
├── sailing_env.py        # Ambiente PettingZoo (logica di gioco, reward, render)
├── wind_model.py         # Campo di vento 2D con random walk (modulo separato)
├── callbacks.py          # Callback SB3: traccia success rate e distanza media
│
├── models/               # Modelli salvati (.zip) — NON su git
├── videos/               # Video output (.mp4) — NON su git
├── sailing_tensorboard/  # Log TensorBoard — NON su git
│
├── requirements.txt      # Dipendenze Python
├── .env                  # Variabili d'ambiente locali — NON su git
└── .gitignore
```

---

## Branch attive

| Branch | Descrizione |
|--------|-------------|
| `main` | Codice stabile, funzionante |
| `feature/pettingzoo-single-agent` | Migrazione ambiente a PettingZoo (singolo agente) |
| `feat-pettingzoo-migration` | Migrazione completa multi-agent PettingZoo |
| `samir/pettingzoo-single-agent` | Sviluppo personale: vento random walk, miglioramenti env |

---

## Roadmap di sviluppo

### Fase 1 — Singolo agente (in corso)

- [x] Ambiente base PettingZoo (`sailing_env.py`)
- [x] Training PPO con SuperSuit (`train_ppo.py`)
- [x] Generazione video (`evaluate_ppo.py`)
- [x] Callback metriche (`callbacks.py`)
- [x] **Campo di vento 2D con random walk** (`wind_model.py`)
- [ ] Waypoint/gate multipli (percorso di regata: bolina → boa → poppa → arrivo)
- [ ] Azioni continue: angolo timone, trim vele, foil up/down
- [ ] Gestione foil (decollo/atterraggio in base a velocità)
- [ ] Fase pre-partenza (countdown, entry timing, penalità anticipo)
- [ ] Reward shaping: VMG (Velocity Made Good), efficienza virate
- [ ] Rendering migliorato: frecce vento, gate, indicatore foil

### Fase 2 — Multi-agent (da iniziare dopo Fase 1)

- [ ] Aggiungere `boat_1` in `sailing_env.py`
- [ ] Osservazione relativa dell'avversario nell'observation space
- [ ] Regole: diritto di precedenza (mura destra/sinistra), boundary box, penalità collisione
- [ ] Reward competitivo (vantaggio tattico, copertura vento)
- [ ] Self-play training
- [ ] Metriche separate per agente in `callbacks.py`

---

## Note per i collaboratori

- **Non pushare mai** `.env`, `.venv/`, `models/`, `videos/` — sono in `.gitignore`
- Prima di iniziare a lavorare: `git pull` e `source .venv/bin/activate`
- Ogni nuova funzionalità va su una branch dedicata (`feature/nome-funzionalità`)
- Per aggiungere dipendenze: aggiornare anche `requirements.txt` con `pip freeze > requirements.txt` (o aggiungere manualmente solo le nuove)
