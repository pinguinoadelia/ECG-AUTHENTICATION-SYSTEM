import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def _extract_fiducials(ecg, fs, qrs_height, p_height, t_height):
    qrs_peaks, _ = find_peaks(ecg, distance=int(0.3*fs), height=qrs_height)
    p_peaks = []
    for r in qrs_peaks:
        start = max(0, r - int(0.2*fs))
        peaks, _ = find_peaks(ecg[start:r], height=p_height)
        if peaks.size:
            p_peaks.append(start + peaks[-1])
    t_peaks = []
    for r in qrs_peaks:
        end = min(len(ecg), r + int(0.4*fs))
        peaks, _ = find_peaks(ecg[r:end], height=t_height)
        if peaks.size:
            t_peaks.append(r + peaks[0])
    return (
        np.array(p_peaks, dtype=int),
        np.array(qrs_peaks, dtype=int),
        np.array(t_peaks, dtype=int)
    )


def plot_svm_fiducials(ecg_segmento, title, fs=360):
    p_peaks, qrs_peaks, t_peaks = _extract_fiducials(
        ecg_segmento, fs,
        qrs_height=0.5, p_height=0.1, t_height=0.1
    )
    plt.figure(figsize=(12,4))
    plt.plot(ecg_segmento, label='ECG')
    plt.scatter(qrs_peaks, ecg_segmento[qrs_peaks], marker='o', color='red',   label='QRS (SVM)')
    plt.scatter(p_peaks,   ecg_segmento[p_peaks],   marker='^', color='green', label='P (SVM)')
    plt.scatter(t_peaks,   ecg_segmento[t_peaks],   marker='v', color='magenta',label='T (SVM)')
    plt.title(f"SVM {title}")
    plt.xlabel("Campioni")
    plt.ylabel("Ampiezza")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cnn_lstm_fiducials(ecg_segmento, title, fs=360):
    p_peaks, qrs_peaks, t_peaks = _extract_fiducials(
        ecg_segmento, fs,
        qrs_height=0.4, p_height=0.08, t_height=0.08
    )
    plt.figure(figsize=(12,4))
    plt.plot(ecg_segmento, label='ECG')
    plt.scatter(qrs_peaks, ecg_segmento[qrs_peaks], marker='D', color='blue',  label='QRS (DL)')
    plt.scatter(p_peaks,   ecg_segmento[p_peaks],   marker='<', color='yellow',label='P (DL)')
    plt.scatter(t_peaks,   ecg_segmento[t_peaks],   marker='x', color='black', label='T (DL)')
    plt.title(f"CNN-LSTM {title}")
    plt.xlabel("Campioni")
    plt.ylabel("Ampiezza")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
