# Forex PCA & Network Analysis Guide

Questo documento spiega come interpretare i grafici generati dall'analisi delle Componenti Principali (PCA) e delle correlazioni residue sull'universo Forex (27 coppie, dati a 1 minuto).

---

### 01. Explained Variance (`01_explained_variance.png`)
**Cosa rappresenta:** Mostra quanta informazione (varianza) del mercato Forex è spiegata da ogni singola "Componente Principale" (PC).
*   **Barre (Individuali):** La % di varianza spiegata da quel fattore specifico. Solitamente la PC1 domina perché rappresenta il "Dollar Factor" o il "Market Mode".
*   **Linea (Cumulativa):** La somma della varianza spiegata man mano che aggiungiamo componenti.
*   **Interpretazione:** Se la PC1 spiega, ad esempio, il 40%, significa che quasi metà dei movimenti di 27 coppie diverse è dovuta a un unico driver globale. Se la pendenza è piatta, il mercato è frammentato e ogni valuta segue la sua strada.

### 02. Loadings Map (`02_loadings_map.png`)
**Cosa rappresenta:** Una mappa geografica-finanziaria delle coppie. Posiziona ogni coppia in base alla sua sensibilità ai primi due fattori (PC1 e PC2).
*   **Asse X (PC1):** Solitamente la forza/debolezza del Dollaro. Le coppie sulla destra si muovono insieme contro quelle sulla sinistra.
*   **Asse Y (PC2):** Spesso rappresenta il sentiment di rischio (Risk On/Off) o un blocco regionale (es. Euro-zone).
*   **Interpretazione:** 
    *   **Vicinanza:** Coppie vicine nel grafico sono altamente correlate (es. AUDUSD e NZDUSD).
    *   **Opposizione:** Coppie in quadranti opposti tendono a muoversi in direzioni diverse.
    *   **Isolamento:** Una coppia lontana da tutte le altre ha driver unici (ottima per diversificare).

### 03. Rolling Synchronization (`03_rolling_synchronization.png`)
**Cosa rappresenta:** L'evoluzione temporale dell'importanza della PC1 su una finestra mobile di 4 giorni.
*   **Picchi:** Indicano momenti di **alta sincronizzazione**. Durante crisi o forti trend del Dollaro, tutte le coppie iniziano a muoversi insieme. La Mean Reversion qui è pericolosa perché i movimenti sono direzionali e sistemici.
*   **Valli:** Indicano momenti di **decoupling**. Le valute si muovono per motivi locali (idiosincratici). È l'ambiente ideale per la Mean Reversion e il trading di cross-pair.

### 04. Systemic Exposure (`04_systemic_exposure.png`)
**Cosa rappresenta:** La "fedeltà" di ogni coppia al fattore globale (PC1). Tecnicamente è l'R-quadro tra la coppia e la PC1.
*   **Barre Alte (80%+):** "Slave of the Market". La coppia non ha quasi vita propria, segue pedissequamente il driver globale (es. EURUSD rispetto al Dollaro).
*   **Barre Basse (<30%):** "Lone Wolves". La coppia è guidata da fattori locali. Queste sono le coppie più interessanti per trovare Alpha che non sia solo una scommessa sul Dollaro.

### 05. Residual Network (`05_residual_network.png`)
**Cosa rappresenta:** La **vera influenza reciproca**, pulita dal rumore del Dollaro. È una heatmap di correlazione calcolata sui *residui* dopo aver rimosso la PC1.
*   **Quadrati Rossi Intensi:** Fortissima correlazione positiva "nascosta". Esempio classico: AUD e NZD. Anche se togli il Dollaro, restano legate perché le loro economie sono gemelle.
*   **Quadrati Blu Intensi:** Correlazione negativa nascosta. 
*   **Zone Bianche:** Coppie che non hanno alcuna relazione diretta tra loro una volta rimosso l'effetto del mercato globale.
*   **Uso:** Fondamentale per il **Pairs Trading**. Se due coppie hanno un residuo molto correlato, puoi tradare l'una contro l'altra sapendo che il legame è strutturale e non dovuto al Dollaro.
