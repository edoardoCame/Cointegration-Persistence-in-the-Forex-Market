# Guida all'Interpretazione: Cross-Asset Predictive Network

Questo grafo visualizza la struttura delle influenze predittive tra diverse coppie valutarie. L'analisi si basa sulla capacità dei rendimenti passati (**Lookback 60 min**) di un asset di prevedere i rendimenti futuri (**Horizon 15 min**) di un altro asset.

## 1. I Nodi (Le Valute)

### Colore: Leadership (Out-Degree)
Il colore del nodo indica quanto quell'asset è un **"Leader"** del mercato:
*   **Viola/Blu Scuro (Alto Out-Degree):** Sono i motori del segnale. I movimenti passati di queste valute influenzano il futuro di molti altri asset. Sono generatori di alpha cross-asset.
*   **Giallo/Verde Chiaro (Basso Out-Degree):** Sono asset "isolati" o "terminali" dal punto di vista predittivo; non proiettano segnali forti verso gli altri.

### Dimensione: Prevedibilità (In-Degree)
La grandezza del nodo indica quanto quell'asset è **"Prevedibile"**:
*   **Nodi Grandi:** Ricevono molte frecce. Il loro andamento futuro è fortemente influenzato (correlato) a ciò che è successo 60 minuti prima su altri asset. Sono ottimi "target" per strategie cross-asset.
*   **Nodi Piccoli:** Sono più indipendenti o guidati da dinamiche proprie non catturate da questo modello.

## 2. Le Frecce (I Link Predittivi)

### Direzione: Chi guida chi
*   La freccia parte dal **Predictor** (passato) e punta al **Target** (futuro).
*   Se vedi una freccia da EURUSD a GBPUSD, significa che il movimento di EURUSD nell'ultima ora aiuta a prevedere cosa farà GBPUSD nei prossimi 15 minuti.

### Colore: Tipo di Relazione
*   **Verde (Correlazione Positiva):** **Momentum Cross-Asset**. Se il Predictor sale, è probabile che il Target salga nei prossimi 15 min.
*   **Rosso (Correlazione Negativa):** **Mean Reversion Cross-Asset**. Se il Predictor sale, è probabile che il Target scenda (o inverta) nei prossimi 15 min.

### Spessore e Opacità: Forza del Segnale
*   **Linee Spesse e Accese:** Rappresentano le connessioni statisticamente più forti (Top 50 in assoluto nel mercato). Sono i legami più affidabili su cui costruire un modello di trading.
*   **Linee Sottili e Sfumate:** Sono connessioni reali ma più deboli rispetto alle dominanti.

## 3. Applicazione Pratica
Cerca i **nodi grandi e viola**: sono i "centri nevralgici" dove il mercato scambia informazioni in modo più efficiente. Un'alta densità di frecce rosse indica un mercato dominato da correzioni e bilanciamenti (Mean Reversion), mentre molte frecce verdi indicano flussi di tendenza che si propagano da una valuta all'altra.
