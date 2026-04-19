# Report confronto: branch dynamic vs progetto locale

Data analisi: 2026-04-18
Branch locale analizzata: samir/pettingzoo-single-agent
Branch target confronto: origin/feat/dynamics-optimization

## 1) Verifiche eseguite

Sono state eseguite queste verifiche Git e di contenuto:

- Stato repository locale (`git status --short --branch`)
- Aggiornamento riferimenti remoti (`git fetch --all --prune`)
- Verifica branch disponibili (`git branch -a`)
- Divergenza commit (`git rev-list --left-right --count HEAD...origin/feat/dynamics-optimization`)
- Merge base comune (`git merge-base HEAD origin/feat/dynamics-optimization`)
- Commit esclusivi locale e dynamic (`git log ...`)
- Differenze file e volume modifiche (`git diff --name-status`, `--numstat`, `--stat`, `--summary`)
- Analisi diff dettagliata su file chiave:
  - `train_ppo.py`
  - `main.py`
  - `evaluate_ppo.py`
  - `callbacks.py`
  - `requirements.txt`
  - `README.md`
- Ispezione nuovi file introdotti in dynamic:
  - `config.yaml`
  - `env_sailing_env.py`
  - `core_boat_physics.py`
  - `core_sail_trim.py`
  - `core_wind_model.py`
  - `env_rendering.py`
  - `scratch_eval.py`
  - `test_eval_debug.py`

Nota: non sono stati modificati file di codice durante questa analisi.

## 2) Risultato sintetico divergenza branch

- Divergenza commit (`HEAD...origin/feat/dynamics-optimization`):
  - Locale avanti: 1 commit
  - Dynamic avanti: 14 commit
- Merge base comune: `f01a6bf848dba217d30d0c6760c0085522ac4a6c`

Commit solo locale:
- `d17eb14` Keep useful PPO and rounding changes

Commit solo dynamic (ordine recente -> meno recente):
- `9a396df` feat: reward changes
- `7c804a8` fix: bug in loop avoidance
- `ac75190` fix: fixing elusion starting line and asymmetric collisions
- `ebdf49a` fix: video generation bug fixed
- `a5f7d2a` feat: enlarged model, fix: treining video generation
- `eb59887` feat: removed unused checkpoints, created yaml for easier commands, updated readme
- `029f18b` rimozione video non necessario
- `4c69d15` fix: fix di bug e warnings nel codice, rimozione di modelli non più utilizzabili
- `267f982` Update train with saving, -inf reward for collisions/boundaries, and new map
- `776cc7a` Update dynamics and rendering from dynamics-optimization
- `ab7a102` Refactor sailing env into core modules & renderer
- `6a44df6` collissioni

## 3) Differenze file (elenco completo)

Output principale (`git diff --name-status HEAD...origin/feat/dynamics-optimization`):

- D `.env.example`
- M `README.md`
- M `callbacks.py`
- A `config.yaml`
- A `core_boat_physics.py`
- A `core_sail_trim.py`
- A `core_wind_model.py`
- A `env_rendering.py`
- A `env_sailing_env.py`
- M `evaluate_ppo.py`
- M `main.py`
- A `models/sailing_ac_new.zip`
- D `models/sailing_ppo_improved.zip`
- D `models/sailing_ppo_realistic_until100.zip`
- M `requirements.txt`
- D `sail_trim.py`
- D `sailing_env.py`
- A `scratch_eval.py`
- A `test_eval_debug.py`
- M `train_ppo.py`
- D `wind_model.py`

Volume modifiche (`--stat`):
- 21 file cambiati
- 2001 inserimenti
- 1180 cancellazioni

## 4) Cambiamenti architetturali principali

### 4.1 Rifattorizzazione ambiente

Dynamic sostituisce il monolite ambiente:
- Rimossi:
  - `sailing_env.py`
  - `sail_trim.py`
  - `wind_model.py`
- Introdotti:
  - `env_sailing_env.py` (orchestrazione ambiente)
  - `core_boat_physics.py` (fisica/polare/VMG)
  - `core_sail_trim.py` (logica trim)
  - `core_wind_model.py` (vento)
  - `env_rendering.py` (render separato)

Impatto:
- Chi importa direttamente moduli legacy deve migrare import ai nuovi file.
- Il rendering viene separato in classe dedicata (`SailingRenderer`).

### 4.2 Introduzione configurazione centralizzata YAML

Nuovo file:
- `config.yaml`

Contiene:
- Parametri run (`steps`, `n_envs`, `model_name`, `video_file`)
- Iperparametri training (`learning_rate`, `n_steps`, `batch_size`, `frame_stack`, `success_threshold_pct`, ecc.)
- Parametri fisica (`max_rudder_delta_per_step`, inerzie)

Impatto:
- Necessaria dipendenza PyYAML.
- La branch dynamic dipende da `config.yaml` in vari entrypoint.

### 4.3 Cambio pipeline training

File: `train_ppo.py`

Cambi chiave in dynamic:
- Import ambiente da `env_sailing_env` (non più `sailing_env`).
- Uso `VecFrameStack` (stack osservazioni, default da config: 4).
- Caricamento automatico di modello esistente (`PPO.load`) se `<model_path>.zip` esiste.
- Callback multipla (`CallbackList`) con:
  - `SuccessTrackingCallback`
  - `CleanCheckpointCallback` (nuova)
- Pulizia checkpoint vecchi in `models/checkpoints` quando crea modello nuovo.
- Hyperparametri letti da YAML.

Rischio compatibilità importante:
- Possibile mismatch observation space tra modelli vecchi e nuovo env+frame stack.
- Errore tipico già osservato: model obs dim legacy vs env obs dim stacked.

### 4.4 Cambio entrypoint CLI

File: `main.py`

In dynamic:
- Nuovi flag:
  - `--train-new`
  - `--train-resume`
- Introduzione `--model-name` (base name) invece di `--model-path` esplicito classico.
- Funzione `resolve_model_path` per versioning automatico (`model`, `model_2`, ...).
- Compatibilità modello considera `frame_stack` nella dimensione osservazione.
- In modalità training, genera automaticamente video a fine training (blocco `finally`).
- Soppressione warning SB3/SuperSuit tramite `warnings.filterwarnings`.

Impatto:
- Cambia il flusso operativo rispetto al tuo setup attuale.
- Alcuni comandi CLI precedenti non sono 1:1 equivalenti.

### 4.5 Cambio pipeline video/evaluation

File: `evaluate_ppo.py`

In dynamic:
- Import ambiente da `env_sailing_env`.
- Passaggio a pipeline vettorizzata SuperSuit/VecEnv anche in eval.
- Uso `VecFrameStack` in eval coerente con training.
- Rimosso adattatore osservazioni legacy (`_adapt_obs_for_model`).
- `create_multi_video` passa da 3 a 5 video test.

Impatto:
- Migliore coerenza training/eval.
- Minore tolleranza verso modelli legacy con shape osservazione diverse.

### 4.6 Callback e checkpoint

File: `callbacks.py`

Dynamic introduce:
- Nuova classe `CleanCheckpointCallback` per mantenere solo ultimi N checkpoint.
- `SuccessTrackingCallback` esteso con `success_threshold_pct` (non solo 100% se configurato).

Impatto:
- Training più gestibile su disco.
- Condizione di stop più flessibile.

## 5) Dipendenze e file runtime

### 5.1 requirements

Aggiunta in dynamic:
- `PyYAML>=6.0`

### 5.2 file modello nel repository

Differenze binarie:
- Aggiunto: `models/sailing_ac_new.zip`
- Rimossi:
  - `models/sailing_ppo_improved.zip`
  - `models/sailing_ppo_realistic_until100.zip`

Impatto:
- Se ti serve conservare i modelli legacy nel tuo workflow, dynamic li rimuove dal diff rispetto al tuo branch attuale.

### 5.3 `.env.example`

In dynamic risulta eliminato (`D .env.example`).

## 6) Impatti funzionali attesi se applichi dynamic al tuo locale

- Dovrai migrare import da moduli legacy (`sailing_env`, `wind_model`, `sail_trim`) ai nuovi moduli core.
- I training ripresi da vecchi `.zip` possono fallire per mismatch shape osservazioni (specialmente con frame stack = 4).
- La CLI cambia paradigma da `--train`/`--model-path` a `--train-new`/`--train-resume` + `--model-name`.
- `config.yaml` diventa punto unico di configurazione; senza quel file la branch dynamic usa fallback parziali ma non è equivalente al vecchio flusso.
- Eval/video è più allineata a training vettorizzato ma meno retrocompatibile.

## 7) Piano consigliato di adozione (senza applicare nulla ora)

Ordine suggerito se decidi di integrare dynamic:

1. Portare prima `config.yaml` + `requirements.txt` (PyYAML).
2. Portare i nuovi moduli core e renderer senza toccare ancora l'entrypoint.
3. Migrare `env_sailing_env.py` e poi aggiornare import in `train_ppo.py`/`evaluate_ppo.py`.
4. Migrare `callbacks.py` e il nuovo sistema checkpoint.
5. Migrare `main.py` e validare i nuovi comandi CLI.
6. Decidere strategia modelli:
   - Conservare modelli legacy fuori repo, oppure
   - Rigenerare nuovi modelli coerenti con frame stack/config dynamic.
7. Eseguire test smoke:
   - avvio training corto
   - generazione video singolo
   - generazione multi video

## 8) Conclusione

La branch dynamic non è una piccola patch: è una rifattorizzazione ampia (architettura env, training pipeline, CLI, configurazione, gestione checkpoint, compatibilità modelli).

Integrazione consigliata: graduale e controllata, con focus sulla compatibilità observation space dei modelli esistenti.
