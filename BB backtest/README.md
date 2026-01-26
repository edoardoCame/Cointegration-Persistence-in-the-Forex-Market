# Bollinger Bands (BB) GPU Backtest Engine

Questo modulo fornisce un framework ad alte prestazioni per il backtesting e l'ottimizzazione di strategie basate sulle **Bollinger Bands**, interamente accelerato via GPU utilizzando **RAPIDS (cuDF)**, **Numba CUDA** e **CuPy**.

Il sistema è progettato per gestire grandi moli di dati (M1) e validare la robustezza delle strategie attraverso tecniche avanzate di simulazione.

## 🚀 Caratteristiche Principali

- **GPU-Accelerated Grid Search**: Ottimizzazione parallela dei parametri (Lookback e Std Dev Multiplier) tramite kernel CUDA custom.
- **Walk Forward Analysis (WFA)**: Validazione out-of-sample dinamica per misurare la persistenza delle performance nel tempo.
- **Monte Carlo Simulations**: Stress-test della strategia su migliaia di path sintetici per valutare il rischio di overfitting e la significatività statistica.
- **Bootstrap Generation**: Generazione di path sintetici (Price Shuffling/Resampling) preservando le caratteristiche statistiche del mercato.
- **Integrazione RAPIDS**: Caricamento ed elaborazione dati ultra-veloce con `cudf`.

## 📁 Struttura del Progetto

```text
BB backtest/
├── scripts/
│   ├── bb_gpu_lib.py        # Kernel CUDA core per l'esecuzione della strategia
│   ├── walk_forward_sim.py  # Script per l'esecuzione della Walk Forward Analysis
│   ├── montecarlo_sim.py    # Script per le simulazioni Monte Carlo
│   └── bootstrap_gen.py     # Generatore di path sintetici per Monte Carlo
├── results/
│   ├── walk_forward/        # Risultati WFA (Equity curve, statistiche)
│   └── montecarlo/          # Risultati MC (Distribuzioni di profitto, p-value)
└── bb_optimization_analysis.ipynb # Notebook per l'analisi visuale dei risultati
```

## 🛠️ Moduli e Logica

### 1. `bb_gpu_lib.py`
Contiene la logica "low-level" della strategia implementata in CUDA. Include:
- `bb_batch_grid_search_kernel`: Esegue migliaia di backtest in parallelo su diverse finestre temporali e combinazioni di parametri.
- `bb_wfa_oos_kernel` / `bb_mc_oos_kernel`: Eseguono la validazione Out-Of-Sample (OOS) sui parametri ottimi trovati durante il training.

### 2. `walk_forward_sim.py`
Esegue un ciclo di ottimizzazione su finestre temporali scorrevoli (es. 12 settimane di training, 1 di test). 
- Calcola i migliori parametri per ogni finestra.
- Aggrega i risultati OOS per generare una equity curve realistica.

### 3. `montecarlo_sim.py` & `bootstrap_gen.py`
Il generatore crea file Parquet contenenti centinaia/migliaia di varianti del dataset originale. Lo script di simulazione testa la strategia su questi path per determinare se il profitto ottenuto sui dati reali è frutto del caso o di un reale vantaggio statistico.

## 🚦 Requisiti

- NVIDIA GPU (Pascal o superiore)
- Driver CUDA compatibili
- **RAPIDS AI** (cuDF)
- **Numba**
- **CuPy**

## 📈 Esempio di Utilizzo

Per eseguire una Walk Forward Analysis:
```bash
python "BB backtest/scripts/walk_forward_sim.py"
```

Per generare i path e lanciare una simulazione Monte Carlo:
```bash
python "BB backtest/scripts/bootstrap_gen.py"
python "BB backtest/scripts/montecarlo_sim.py"
```

---
*Nota: Assicurarsi che i dati in formato CSV siano presenti nella cartella `data/` del progetto.*
