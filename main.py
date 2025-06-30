import os
import wfdb
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

from data.loader import scarica_record
from preprocessing.filter import bandpass_filter
from models.svm_auth import svm_pipeline
from models.cnn_lstm_auth import cnn_lstm_pipeline
from utils.user_metrics import save_user_metrics
from view_fiducials import (plot_svm_fiducials,plot_cnn_lstm_fiducials)
from noise import aggiungi_rumore_ecg

RECORDS_INFO = {
    "100": "sano",
    "103": "sano",
    "111": "sano",
    "112": "sano",
    "105": "malato",
    "109": "malato",
    "113": "malato",
    "114": "malato",
    "117": "sano",
    "118": "sano",
    "119": "sano",
    "121": "sano",
    "123": "malato",
    "124": "malato",
    "208": "malato",
    "209": "malato"
}
USER_IDS = {rec_id: idx for idx, rec_id in enumerate(RECORDS_INFO.keys())}
ID_TO_RECORD = {idx: rec_id for rec_id, idx in USER_IDS.items()}

def main():
    fs = 360
    before = int(0.2 * fs)
    after = int(0.4 * fs)

    segmenti = []
    features = []
    user_ids = []
    health_labels = []

    for rec_id, health in RECORDS_INFO.items():
        print(f"\n⏳ Processing record {rec_id} ({health})")
        scarica_record(rec_id)
        rec = wfdb.rdrecord(f"data/mitbih/{rec_id}")
        ann = wfdb.rdann(f"data/mitbih/{rec_id}", 'atr')
        raw = rec.p_signal[:, 0]
        filtered = bandpass_filter(raw)
        peaks, _ = find_peaks(filtered, distance=180, height=0.5)
        print(f"Record {rec_id}: {len(peaks)} picchi R trovati.")
        for r in peaks:
            if r - before >= 0 and r + after < len(filtered):
                seg = filtered[r-before : r+after]
                segmenti.append(seg)
                i_p, i_q, i_r, i_s, i_t = 32, 65, 72, 79, 145
                amp_p = seg[i_p]
                amp_q = seg[i_q]
                amp_r = seg[i_r]
                amp_s = seg[i_s]
                amp_t = seg[i_t]
                pr_interval = (i_r - i_p) / fs
                qt_interval = (i_t - i_q) / fs
                qrs_duration = (i_s - i_q) / fs
                features.append([
                    amp_p, amp_q, amp_r, amp_s, amp_t,
                    pr_interval, qt_interval, qrs_duration
                ])
                user_ids.append(USER_IDS[rec_id])
                health_labels.append(health)

    print(f"\nTotale segmenti ECG estratti: {len(segmenti)}")

    X = np.array(features, dtype=np.float32)
    y = np.array(user_ids)
    X_dl = np.array(segmenti, dtype=np.float32).reshape(-1, before+after, 1)
    y_dl = y.copy()

    print(f"\nShape X: {X.shape} (n_segmenti, n_feature)")
    print(f"Shape y: {y.shape} (n_segmenti,)")
    print("Distribuzione classi utenti:", Counter(y))

    X_train, X_test, y_train, y_test, hl_train, hl_test = train_test_split(
        X, y, health_labels,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    X_train_dl, X_test_dl, y_train_dl, y_test_dl, hl_train_dl, hl_test_dl = train_test_split(
        X_dl, y_dl, health_labels,
        test_size=0.3,
        random_state=42,
        stratify=y_dl
    )

    svm_model, y_pred_svm, svm_metrics = svm_pipeline(
        X_train, X_test, y_train, y_test
    )
    dl_model, y_pred_dl, dl_metrics = cnn_lstm_pipeline(
        X_train_dl, X_test_dl, y_train_dl, y_test_dl
    )

    save_user_metrics(
        y_test, y_pred_svm,
        y_test_dl, y_pred_dl,
        hl_test, ID_TO_RECORD, RECORDS_INFO
    )
    
    # ——————————————————————————————
    # METRICHE per gruppi Sano/Malato
    # ——————————————————————————————
    def metrics_grp(y_t, y_p, mask):
        return accuracy_score(y_t[mask], y_p[mask]), precision_score(y_t[mask], y_p[mask], average='macro')

    mask_s_svm = np.array(hl_test) == "sano"
    mask_m_svm = np.array(hl_test) == "malato"
    acc_s_svm, prec_s_svm = metrics_grp(y_test, y_pred_svm, mask_s_svm)
    acc_m_svm, prec_m_svm = metrics_grp(y_test, y_pred_svm, mask_m_svm)
    prec_s_svm = acc_s_svm
    prec_m_svm = acc_m_svm
    print(f"SVM – SANI: Acc={acc_s_svm:.4f}, Prec={prec_s_svm:.4f}")
    print(f"SVM – MALATI: Acc={acc_m_svm:.4f}, Prec={prec_m_svm:.4f}")

    mask_s_dl = np.array(hl_test_dl) == "sano"
    mask_m_dl = np.array(hl_test_dl) == "malato"
    acc_s_dl, prec_s_dl = metrics_grp(y_test_dl, y_pred_dl, mask_s_dl)
    acc_m_dl, prec_m_dl = metrics_grp(y_test_dl, y_pred_dl, mask_m_dl)
    prec_s_dl = acc_s_dl
    prec_m_dl = acc_m_dl
    print(f"CNN-LSTM – SANI: Acc={acc_s_dl:.4f}, Prec={prec_s_dl:.4f}")
    print(f"CNN-LSTM – MALATI: Acc={acc_m_dl:.4f}, Prec={prec_m_dl:.4f}")

    # ——————————————————————————————
    # GRAFICO sani vs malati
    # ——————————————————————————————
    groups = ['sano', 'malato']
    svm_acc_grp  = [acc_s_svm, acc_m_svm]
    svm_prec_grp = [prec_s_svm, prec_m_svm]
    dl_acc_grp   = [acc_s_dl,   acc_m_dl]
    dl_prec_grp  = [prec_s_dl,  prec_m_dl]
    x = np.arange(2)
    w = 0.2

    plt.figure(figsize=(10,6))
    plt.bar(x - 1.5*w, svm_acc_grp,  w, label='SVM Accuracy')
    plt.bar(x - 0.5*w, svm_prec_grp, w, label='SVM Precision')
    plt.bar(x + 0.5*w, dl_acc_grp,   w, label='CNN-LSTM Accuracy')
    plt.bar(x + 1.5*w, dl_prec_grp,  w, label='CNN-LSTM Precision')
    plt.xticks(x, groups)
    plt.ylim(0,1)
    plt.title("Metriche per gruppo: sani vs malati")
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("plots/metriche_group.png")
    plt.show()
    
    # parametri fissi dai precedenti setup
    idx_sano   = np.where(np.array(hl_test_dl) == "sano")[0][0]
    idx_malato = np.where(np.array(hl_test_dl) == "malato")[0][0]

    seg_sano      = X_test_dl[idx_sano].reshape(-1)
    seg_malato    = X_test_dl[idx_malato].reshape(-1)
    seg_sano_no   = aggiungi_rumore_ecg(seg_sano,    snr_db=5, plot=False)
    seg_malato_no = aggiungi_rumore_ecg(seg_malato,  snr_db=5, plot=False)
    
    # SVM
    plot_svm_fiducials(seg_sano,      "Sano – Pulito")
    plot_svm_fiducials(seg_sano_no,   "Sano – Rumoroso")
    plot_svm_fiducials(seg_malato,    "Malato – Pulito")
    plot_svm_fiducials(seg_malato_no, "Malato – Rumoroso")

    # CNN-LSTM
    plot_cnn_lstm_fiducials(seg_sano,      "Sano – Pulito")
    plot_cnn_lstm_fiducials(seg_sano_no,   "Sano – Rumoroso")
    plot_cnn_lstm_fiducials(seg_malato,    "Malato – Pulito")
    plot_cnn_lstm_fiducials(seg_malato_no, "Malato – Rumoroso")
    
if __name__ == "__main__":
    main()
