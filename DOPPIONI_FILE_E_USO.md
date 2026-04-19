# File duplicati e come usarli

Nel progetto convivono alcuni file "originali" e alcune versioni "dynamic" o modulari. Questa scelta serve a mantenere stabile la versione già usata per training e video, mentre si sperimentano nuove strutture di codice senza rompere il flusso storico.

## Perché esistono i doppioni

I doppioni sono stati creati per separare due obiettivi:

- mantenere i file storici funzionanti e confrontabili;
- introdurre una versione modulare senza sovrascrivere il comportamento precedente;
- rendere più semplice il confronto tra implementazione locale e implementazione dinamica.

## Coppie principali

- [main.py](main.py) e [main_dynamic.py](main_dynamic.py)
- [train_ppo.py](train_ppo.py) e [train_ppo_dynamic.py](train_ppo_dynamic.py)
- [evaluate_ppo.py](evaluate_ppo.py) e [evaluate_ppo_dynamic.py](evaluate_ppo_dynamic.py)
- [sailing_env.py](sailing_env.py) e [env_sailing_env.py](env_sailing_env.py)

## Cosa usare in pratica

### Versione originale

Usa i file originali se vuoi:

- ripetere il flusso storico del progetto;
- confrontare i risultati con i modelli e le osservazioni precedenti;
- mantenere il comportamento già validato dalla versione locale.

Comandi tipici:

```bash
.venv/bin/python main.py --train
.venv/bin/python main.py --model-path models/sailing_ppo_realistic_until100 --video-file videos/sailing_realistic_until100.mp4
```

### Versione dynamic

Usa i file dynamic se vuoi:

- lavorare sulla versione modulare e più scomposta del progetto;
- usare l'entrypoint separato senza toccare il flusso originale;
- testare le modifiche nuove in modo isolato.

Comandi tipici:

```bash
.venv/bin/python main_dynamic.py --train
.venv/bin/python main_dynamic.py --model-path models/sailing_ppo_rounding_full_v10 --video-file videos/sailing_ppo_rounding_full_v10.mp4
```

## File di supporto aggiunti con la versione modularizzata

Questi file aiutano a spezzare la logica in moduli più piccoli:

- [core_boat_physics.py](core_boat_physics.py)
- [core_sail_trim.py](core_sail_trim.py)
- [core_wind_model.py](core_wind_model.py)
- [env_rendering.py](env_rendering.py)
- [config.yaml](config.yaml)
- [REPORT_CONFRONTO_BRANCH_DYNAMIC_VS_LOCALE_2026-04-18.md](REPORT_CONFRONTO_BRANCH_DYNAMIC_VS_LOCALE_2026-04-18.md)

## Regola pratica

Se stai solo eseguendo training o video, scegli una sola famiglia di file e non mischiare entrypoint diversi nello stesso test. In generale:

- originale per stabilità e confronto;
- dynamic per evoluzione e modularità.

## Nota sui modelli

I modelli salvati in `models/` possono dipendere dalla struttura delle osservazioni e dalle regole dell'ambiente. Se cambi ambiente o spazio osservazioni, verifica sempre che il modello sia compatibile prima di riusarlo.
