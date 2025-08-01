# ECG Authentication System

Autenticazione biometrica basata su segnali ECG (elettrocardiogramma) acquisiti dal **MIT-BIH Arrhythmia Database**.  
Il sistema confronta due approcci:

* **Support Vector Machine (SVM)** con feature fiduciali estratte manualmente  
* **Deep-Learning 1-D CNN + LSTM** end-to-end sui segmenti grezzi

Entrambe le pipeline producono metriche di accuratezza, FAR/FRR ed EER, generano matrici di confusione e salvano una tabella finale con l’esito (“Successo / Insuccesso”) di autenticazione per ogni utente.

---

## Come funziona

| Fase | Dettagli |
|------|----------|
| **1. Download dati** | `data/loader.py` richiama `wfdb.dl_database` per scaricare i record MIT-BIH elencati in `main.py` (sia soggetti “sani” che “malati”). |
| **2. Pre-processing** | Filtro Butterworth passa-banda 0.5–40 Hz (ordine 2) sugli ECG grezzi. |
| **3. Segmentazione** | Per ogni picco R (rilevato con `scipy.signal.find_peaks`) si estrae una finestra di 0.2 s pre-R + 0.4 s post-R (fs = 360 Hz). |
| **4-A. Feature Engineering (SVM)** | Calcolo manuale delle features con ampiezze delle onde P e T e complesso QRS (con picco in R) e poi classificazione con SVM. |
| **4-B. Deep Learning (CNN-LSTM)** | Architettura: Conv1D(32) → MaxPool → Conv1D(64) → MaxPool → LSTM(64) → Dense; EarlyStopping su `val_loss`. |
| **5. Metriche** | Accuracy, FAR, FRR, EER su test-set stratificato al 30 %; confusion matrix salvata in `plots/`. |
| **6. Output finale** | CSV `results/user_metrics.csv` con esito "Successo" o "Insuccesso" di autenticazione da parte di entrambi i modelli e lo stato di salute "sano" o "malato".|

---

## Avvio rapido

```bash
python main.py
```

Alla prima esecuzione verranno scaricati ≈ 45 MB di dati in `data/mitbih/`.

**Output principali**

- `plots/confusion_matrix_svm.png`
- `plots/confusion_matrix_dl.png`
- `plots/metriche_svm.png`
- `plots/metriche_dl.png`
- `plots/metriche_group.png`
- `results/metriche_comparate.txt`
- `results/user_metrics.csv`
