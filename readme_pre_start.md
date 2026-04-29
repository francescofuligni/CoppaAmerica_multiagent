OBIETTIVO:
Il pre-start serve a simulare la fase prima della partenza di una regata, in cui le barche devono posizionarsi vicino alla linea di partenza, restare in movimento, cercare il timing corretto del via, evitare la partenza anticipata (OCS) e prepararsi all’accelerazione iniziale della gara.

IMPLEMENTAZIONE:

SPOWN BARCHE:
All’inizio di ogni episodio, le due barche vengono posizionate in modo casuale ma controllato per garantire condizioni di partenza eque e realistiche.

Entrambe le barche partono dalla stessa fascia verticale, quindi alla stessa distanza dalla linea di partenza. Questo assicura che nessuna delle due abbia un vantaggio iniziale in termini di avanzamento verso il primo obiettivo.

La posizione orizzontale, invece, è variabile: ciascuna barca viene collocata in un punto diverso lungo la larghezza del campo, evitando però di trovarsi troppo vicina ai bordi.

Inoltre, viene imposto che le due barche non possano partire troppo vicine tra loro, così da evitare collisioni o interferenze immediate all’inizio della regata.

Questo sistema permette di mantenere un buon equilibrio tra casualità e fairness, garantendo allo stesso tempo situazioni iniziali sempre leggermente diverse per favorire un apprendimento più robusto dell’agente.

STATO PRESTART
self.start_phase = True
self.start_armed = False
self.start_timer = 40

start_phase indica che il sistema è nella fase pre-start.
start_timer definisce la durata della fase.
start_armed diventa True quando la gara è ufficialmente attiva.

REWARD PRESTART
Attivo solo quando self.start_phase == True.
Vicinanza alla linea di partenza:
reward += (1.0 - dist_to_line / 200.0) * 3.0
→ premia stare vicino alla linea
Avanzamento del tempo:
reward += timer_ratio * 2.0
→ premia l’avvicinarsi del momento del via
Velocità:
reward += speed * 0.1
→ premia il movimento e l’accelerazione
OCS (FALSE START)
if self.state[agent]['y'] > self.start_line_y + 2.0:
reward -= self.ocs_penalty
terminated = True

Se la barca supera la linea prima del via:
→ penalità forte
→ terminazione episodio (squalifica simulata)

FINE PRESTART
if self.step_count >= self.start_timer:
self.start_phase = False
self.start_armed = True

Alla fine del timer:
→ il prestart termina
→ inizia la fase di regata vera

POST START (GARA NORMALE):

Quando start_phase diventa False:

scompaiono tutti i reward legati alla linea
scompare il reward temporale
scompare il bonus di posizione
il controllo OCS non viene più applicato

Resta attiva solo la logica di gara:

VMG (velocità verso obiettivo)
gestione vento e velocità
rotta e heading
passaggio boe
collisioni
completamento regata

COMPORTAMENTO ATTESO:

Durante pre-start:

le barche si avvicinano progressivamente alla linea
aumentano la velocità verso la fine del timer
evitano di superare la linea (OCS)
cercano una buona posizione di partenza

Dopo lo start:

cambia completamente il comportamento
il modello passa alla strategia di regata
ottimizzazione su velocità e rotta verso le boe