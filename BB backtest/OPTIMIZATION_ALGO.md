# Algoritmo di Ottimizzazione Bollinger Bands (GPU Accelerated)

L'obiettivo dell'algoritmo è trovare la combinazione ottimale di parametri $(L, \sigma)$ che massimizzi il rendimento netto, minimizzando il rischio di sovra-ottimizzazione tramite validazione Out-of-Sample.

## 1. Modello Matematico della Strategia

La strategia è un sistema **Mean Reverting** basato sulle Bande di Bollinger.

### Indicatori
Dati i prezzi di chiusura $P$, definiamo per ogni istante $t$ e lookback $L$:
- **Media Mobile Semplice:** $\mu_{t,L} = \frac{1}{L} \sum_{i=0}^{L-1} P_{t-i}$
- **Deviazione Standard:** $s_{t,L} = \sqrt{\frac{1}{L} \sum_{i=0}^{L-1} (P_{t-i} - \mu_{t,L})^2}$
- **Upper Band:** $UB_{t,L,\sigma} = \mu_{t,L} + \sigma \cdot s_{t,L}$
- **Lower Band:** $LB_{t,L,\sigma} = \mu_{t,L} - \sigma \cdot s_{t,L}$

### Logica di Ingresso e Uscita
La posizione $S_t \in \{1, 0, -1\}$ (Long, Flat, Short) evolve secondo le seguenti regole:

- **Entry Long:** Se $S_{t-1} = 0$ e $P_t < LB_{t,L,\sigma}$
- **Entry Short:** Se $S_{t-1} = 0$ e $P_t > UB_{t,L,\sigma}$
- **Exit Long:** Se $S_{t-1} = 1$ e $P_t \ge \mu_{t,L}$
- **Exit Short:** Se $S_{t-1} = -1$ e $P_t \le \mu_{t,L}$

## 2. Funzione Obiettivo e Costi

Il profitto di un singolo trade $i$ è calcolato come:
$$R_i = (P_{exit} - P_{entry}) \cdot \text{direction} - (C_{in} + C_{out})$$
Dove $C$ è la commissione in pips (es. 0.6 pips).

L'ottimizzatore massimizza la funzione obiettivo richiesta:

$$
\begin{aligned}
\max_{L, \sigma} \quad & \sum_{i=1}^{N} R_i \\
\text{s.t.} \quad & N_{\text{trades}} \ge N_{\min}
\end{aligned}
$$

*(Nello script, $N_{\min} = 20$ per evitare parametri che generano trade isolati e casuali).* 

## 3. Implementazione GPU (Grid Search Parallela)

L'algoritmo utilizza un approccio **Brute Force Grid Search** parallelizzato tramite CUDA:

1.  **Spazio di Ricerca:** Definiamo un set discreto di Lookbacks $\mathcal{L} = \{1000, 2000, \dots, 20000\}$ e Multipli $\mathcal{S} = \{1.5, 1.6, \dots, 7.0\}$.
2.  **Parallelismo:** Ogni thread sulla GPU è responsabile di una singola coppia $(L, \sigma)$. 
3.  **Path Dependency:** Poiché la strategia dipende dalla posizione precedente ($S_{t-1}$), il thread deve iterare sequenzialmente lungo la serie temporale dei prezzi, accumulando i rendimenti e contando i trade.
4.  **Memoria:** I dati di input (prezzi, medie pre-calcolate, deviazioni) sono caricati nella memoria globale della GPU per un accesso ultra-veloce da parte dei kernel.

## 4. Validazione In-Sample / Out-of-Sample

Per combattere il *Curve Fitting*:
- **In-Sample (70%):** La GPU esplora tutte le combinazioni e identifica $(\hat{L}, \hat{\sigma})$ che massimizzano il rendimento.
- **Out-of-Sample (30%):** I parametri "vincitori" vengono testati su dati mai visti dall'algoritmo. La vicinanza tra la pendenza della curva IS e OOS è il principale indicatore di robustezza.
