# ECG Authentication System

Autenticazione biometrica basata su segnali ECG (elettrocardiogramma) acquisiti dal **MIT-BIH Arrhythmia Database**.  
Il sistema confronta due approcci:

* **Support Vector Machine (SVM)** con feature fiduciali estratte manualmente  
* **Deep-Learning 1-D CNN + LSTM** end-to-end sui segmenti grezzi

Entrambe le pipeline producono metriche di accuratezza, FAR/FRR ed EER, generano matrici di confusione e salvano una tabella finale con l’esito (“Successo / Insuccesso”) di autenticazione per ogni utente.

---

## Contenuti della repository

ECG-AUTHENTICATION-SYSTEM/
│
├─ data/ # downloader e dati MIT-BIH (ignorati dal VCS)
│ └─ loader.py # scarica i record richiesti dal database
├─ preprocessing/
│ └─ filter.py # filtro Butterworth passa-banda 0.5-40 Hz
├─ models/
│ ├─ svm_auth.py # pipeline SVM + metriche + confusion matrix + plots
│ └─ cnn_lstm_auth.py # architettura CNN-LSTM + metriche + confusion matrix + plots
├─ utils/
│ └─ user_metrics.py # genera CSV con esito per utente
├─ plots/ # immagini auto-generate (confusion & metriche)
├─ results/ # output CSV auto-generato
├─ main.py # punto d’ingresso: orchestrazione completo
├─ noise.py # funzione per aggiungere rumore gaussiano al segnale
├─ view_fiducials.py # visualizzazione interattiva dei picchi P-QRS-T
├─ requirements.txt # dipendenze Python
└─ .gitignore

## Come funziona

| Fase | Dettagli |
|------|----------|
| **1. Download dati** | `data/loader.py` richiama `wfdb.dl_database` per scaricare i record MIT-BIH elencati in `main.py` (sia soggetti “sani” che “malati”). |
| **2. Pre-processing** | Filtro Butterworth passa-banda 0.5–40 Hz (ordine 2) sugli ECG grezzi. |
| **3. Segmentazione** | Per ogni picco R (rilevato con `scipy.signal.find_peaks`) si estrae una finestra di 0.2 s pre-R + 0.4 s post-R (fs = 360 Hz). |
| **4-A. Feature Engineering (SVM)** | Calcolo manuale di 8 feature: ampiezze P–Q–R–S–T e intervalli PR, QT, QRS, poi classificazione con SVM. |
| **4-B. Deep Learning (CNN-LSTM)** | Architettura: Conv1D(32) → MaxPool → Conv1D(64) → MaxPool → LSTM(64) → Dense; EarlyStopping su `val_loss`. |
| **5. Metriche** | Accuracy, FAR, FRR, EER su test-set stratificato al 30 %; confusion matrix salvata in `plots/`. |
| **6. Output finale** | CSV `results/user_metrics.csv` con esito (“Successo” se accuracy ≥ 0.85). |

