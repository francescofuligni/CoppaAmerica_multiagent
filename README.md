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

## Configurazione (config.yaml)

Tutti i parametri nevralgici del progetto (passi di training, numero di ambienti, iperparametri RL e costanti fisiche) sono centralizzati nel file `config.yaml`. Questo permette di modificare il setup senza toccare il codice.

```yaml
run:
  steps: 1000000
  n_envs: 14
  model_name: "sailing_model"
  video_file: "videos/sailing_demo.mp4"

training:
  learning_rate: 0.0002
  frame_stack: 4
  # ... altri iperparametri
```

---

## Avvio e Utilizzo

Assicurati di avere l'ambiente virtuale attivo (`source .venv/bin/activate`). Il sistema utilizza un **versionamento automatico** intelligente per i modelli (es. `sailing_model.zip`, `sailing_model_2.zip`).

### 1. Addestramento (Training)

#### Creare un NUOVO Modello
Parte da zero e crea la versione numerata successiva. Pulisce automaticamente i vecchi checkpoint temporanei.
```bash
python main.py --train-new
```

#### Riprendere l'ultimo training
Carica l'ultima versione del modello trovata in `models/` e continua l'addestramento.
```bash
python main.py --train-resume
```

Puoi sovrascrivere i parametri del `config.yaml` direttamente da terminale:
```bash
python main.py --train-new --steps 500000 --n-envs 8
```

### 2. Test e Valutazione

#### Generazione Video Singolo
Usa in automatico l'ultima versione del modello per generare un video della regata.
```bash
python main.py --video-file videos/mio_test.mp4
```

#### Generazione Video Multiplo (5 Seed diversi)
Testa la robustezza del modello su 5 diverse condizioni iniziali/di vento.
```bash
python main.py --test-multi
```

### 3. Monitoraggio (TensorBoard)
Visualizza le curve di apprendimento in tempo reale.
```bash
python -m tensorboard.main --logdir ./sailing_tensorboard/
```

---

## Struttura del progetto

```
CoppaAmerica_multiagent/
│
├── main.py               # Entry point CLI: gestione training (new/resume) e test
├── train_ppo.py          # Logica di training PPO, VecFrameStack e caricamento YAML
├── evaluate_ppo.py       # Valutazione modello e generazione video MP4
├── config.yaml           # Configurazione centralizzata (iperparametri e fisica)
├── callbacks.py          # Callback custom (SuccessTracking e CleanCheckpoint)
├── core/                 # Core della fisica navale e modelli aerodinamici
├── env/                  # Definizione ambiente PettingZoo
│
├── models/               # Modelli salvati (.zip) e cartella checkpoints/
├── videos/               # Output video delle regate
├── sailing_tensorboard/  # Log per la visualizzazione delle metriche
│
├── requirements.txt      # Dipendenze del progetto
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
- [~] Azioni continue: angolo timone, trim vele, foil up/down
- [ ] Gestione foil (decollo/atterraggio in base a velocità)
- [ ] Fase pre-partenza (countdown, entry timing, penalità anticipo)
- [ ] Reward shaping: VMG (Velocity Made Good), efficienza virate
- [ ] Rendering migliorato: frecce vento, gate, indicatore foil

---

## Trim vele (implementato)

Da questa versione il controllo azioni e' continuo su due canali:

- `action[0]`: timone in `[-1, 1]`
- `action[1]`: trim vele in `[-1, 1]` (convertito internamente in livello trim `[0, 1]`)

Dettagli implementativi principali:

- rate limiter sul trim (la vela non puo' cambiare istantaneamente)
- efficienza trim dipendente dal TWA (True Wind Angle)
- penalita' per fuori-trim ad alta velocita'
- tuning VMG esplicito su entrambe le gambe (bolina/poppa)
- fallback automatico per modelli legacy con azione 1D (auto-trim)
- rendering con indicatore `% trim` vicino a ogni barca
- uscita dal campo: nessun rimbalzo; l'episodio termina con `termination_reason='out_of_bounds'`
- successo episodio: valido solo con boa di bolina girata, gate di arrivo, e giro finale (`finished_race`)
- stop training: il callback puo' interrompere automaticamente quando gli ultimi 100 episodi sono tutti successi per entrambi gli agenti

#### Percorso di regata: 3 leg

La regata è strutturata in tre fasi (leg):

1. **Leg 1 (Bolina)**: Sali verso il Top Gate (Y=2300m)
   - Barca parte dal basso (Y≈0-100m) e deve risalire contro il vento
   - Deve attraversare il gate (passando tra le due boe) nella Y ≥ 2300m
   - Gate: ampiezza 300m, centrato nel campo

2. **Leg 2 (Poppa)**: Scendi verso il Bottom Gate (Y=200m)
   - Dopo aver completato Leg 1, la barca gira attorno a una boa
   - Retrocede verso il basso con vento in poppa
   - Deve attraversare il gate della linea di arrivo (Y ≤ 200m)

3. **Leg 3 (Giro finale)**: Giro attorno alla boa di arrivo
   - Dopo aver attraversato il Bottom Gate, la barca deve compiere un giro finale attorno alla boa di arrivo
   - La barca ha un target esterno fissato che la obbliga a fare una curva completa intorno alla boa
   - Durante **Leg 3, la spin violation è disabilitata** perché il giro è una parte legittima della regata
   - Una volta completato il giro (raggiunto il target esterno), l'episodio termina con successo (`finished_race`)

**Implicazioni per l'RL**:
- Il giro del Leg 3 non è una manovra erratica, bensì una fase della regata vera e propria
- Gli agenti imparano a girare in modo controllato attorno alla boa senza essere penalizzati
- La rotazione completa durante il Leg 3 non attiva il vincolo di "spin violation"

### Metriche TensorBoard dedicate al trim

Durante il training vengono loggate anche metriche tecniche aggiuntive:

- `trim/<agent>_efficiency_mean`
- `trim/<agent>_error_mean`
- `vmg/<agent>_mean_kts`
- `speed/<agent>_mean_kts`
- `success/<agent>_rate`

e aggregate globali (`trim/global_*`, `vmg/global_*`, `speed/global_*`, `success/global_rate`).

### Mini test suite (shape + trim)

Esegui i test automatici con:

```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

### Fase 2 — Multi-agent (da iniziare dopo Fase 1)

- [x] Aggiungere `boat_1` in `sailing_env.py`
- [x] Osservazione relativa dell'avversario nell'observation space
- [x] Regole: diritto di precedenza (mura destra/sinistra), boundary box, penalità collisione
- [x] Reward competitivo (vantaggio tattico, copertura vento)
- [x] Self-play training
- [x] Metriche separate per agente in `callbacks.py`

---

## Note per i collaboratori

- **Non pushare mai** `.env`, `.venv/`, `models/`, `videos/` — sono in `.gitignore`
- Prima di iniziare a lavorare: `git pull` e `source .venv/bin/activate`
- Ogni nuova funzionalità va su una branch dedicata (`feature/nome-funzionalità`)
- Per aggiungere dipendenze: aggiornare anche `requirements.txt` con `pip freeze > requirements.txt` (o aggiungere manualmente solo le nuove)

### Novità Fisiche e di Regolamento

- **Cancello di Bolina**: Per completare il primo lato (upwind), le barche devono obbligatoriamente attraversare *in mezzo* alle due boe del cancello prima di aggirarne una per scendere di poppa.
- **Foiling**: Penalità ridotte per caduta dai foil ("drop foil") e per l'abuso di timone, favorendo manovre tattiche e virate ("tack") più strette e realistiche senza che l'agente preferisca uscire dai bordi del campo.

### Aggiornamento Fase 2 (Collisioni e Competizione)

- Collisioni multi-barca con penalità su contatto (`collision_radius`) e separazione simmetrica anti-overlap.
- Penalità di prossimità (`near_collision_radius`) per insegnare avoidance prima dell'impatto.
- Penalità predittiva `time-to-collision` (TTC) quando due traiettorie convergono rapidamente.
- Regola di precedenza semplificata su mure opposte (mura sinistra più penalizzata in collisione).
- Reward competitivo sul distacco tattico tra barche.
- Nuove metriche TensorBoard per collisioni (`collision/*`, `collision/global_*`).

### Hard Violations (Penalità severe)

Il sistema di penalità include un meccanismo di "hard rules" per prevenire comportamenti irrealistici:

- **`boundary_violation`**: Uscita dal campo X (< 500m o > 2000m) → termina episodio con penalità -10,000
- **`out_of_bounds`**: Uscita totale dal campo → termina episodio con penalità -10,000
- **`collision`**: Contatto diretto tra barche (raggio < 20m) → termina episodio con penalità -10,000
- **`missed_top_gate`**: Attraversamento linea top gate (Y ≥ 2300m) fuori dal cancello → termina episodio con penalità -10,000
- **`spin_violation`**: Rotazione > 450° in 30 step senza progresso → termina episodio con penalità -10,000
  - **ECCEZIONE**: Disabilitato durante **Leg 3** (giro finale attorno boa di arrivo), quando la rotazione è parte legittima della regata

**Warmup meccanismo**: Le hard rules hanno un periodo di riscaldamento (`hard_rules_warmup_steps=20,000`) per permettere l'esplorazione iniziale dell'agente senza penalizzazione catastrofica.

### Scelta del Modello: Quale usare?

Nella cartella `models/` troverai vari salvataggi, ma i due principali sono:
- `sailing_ppo_improved.zip`: Modello legacy della versione precedente.
- **`sailing_ppo_realistic_until100.zip` (CONSIGLIATO / DA USARE)**: È l'ultimo modello convergente e il più completo.

**Perché ci sono due modelli e cosa cambia?**
Il modello `realistic_until100` "capisce" le nuove regole geometriche del campo di regata e la nuova fisica. Ecco cosa è stato implementato esattamente in questo branch:
1. **Cancello di Bolina Reale**: Per completare il primo lato (upwind), le barche ora devono obbligatoriamente attraversare *in mezzo* alle due boe del gate prima di aggirarne una e scendere di poppa (nel vecchio modello bastava superare l'asse Y e questo creava bug geometrici).
2. **Foiling Ottimizzato**: Sono state ridotte le penalità catastrofiche per la caduta dal foil e l'uso brusco del timone. L'agente non ha più "paura" di virare, riducendo l'over-standing e favorendo manovre tattiche molto più verosimili.
3. **Addestramento Perfetto**: L'addestramento usa un **Callback (Early Stopping)** customizzato in `callbacks.py` che interrompe il training in automatico solo ed esclusivamente quando le barche chiudono la regata con successo per **100 episodi consecutivi (100% success rate)**.

### Come ri-addestrare (Parametri per i prossimi sviluppi in Multi-Agent)

Se devi implementare nuove feature (ad esempio la visibilità tra barche per permettere l'evitamento collisioni) e hai bisogno di ri-addestrare l'agente, ecco come procedere.

I migliori parametri che hanno portato alla creazione del modello "realistic_until100" sono:
- **Strategia a Chunk (in `train_ppo.py`)**: L'addestramento valuta la rete a intervalli regolari e si ferma da solo se rileva 100 successi in un'unica finestra.
- `total_timesteps=200000` (Spesso converge prima grazie allo stop automatico, tra i 120k e i 150k step).
- `n_envs=12` (Usa quanti più ambienti paralleli riesci. 12 è essenziale per stabilizzare l'apprendimento su CPU, poiché evita che un random walk troppo instabile del vento rompa la policy distruggendo i progressi).

Puoi lanciare il training perfetto direttamente tramite questa istruzione in Python:
```python
from train_ppo import train_model
train_model(total_timesteps=200000, n_envs=12, model_path='models/mio_nuovo_test_collisioni')
```

### Preset PPO (integrazione sicura)

Il training supporta ora due preset PPO per continuare il progetto senza rompere la compatibilità:

- `ppo_preset='safe_optimized'` (default): rollout più lungo e batch più robusto.
- `ppo_preset='legacy'`: mantiene i parametri storici.

Esempio:
```python
from train_ppo import train_model
train_model(
	total_timesteps=200000,
	n_envs=12,
	model_path='models/mio_nuovo_test_collisioni',
	ppo_preset='safe_optimized',
)
```
