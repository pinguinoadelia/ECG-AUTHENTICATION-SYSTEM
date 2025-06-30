import numpy as np
import matplotlib.pyplot as plt

def aggiungi_rumore_ecg(segmento, snr_db=10, plot=True):
    seg = segmento.copy()
    potenza_segnale = np.mean(seg**2)
    snr = 10**(snr_db/10)
    potenza_rumore = potenza_segnale / snr
    rumore = np.random.normal(0, np.sqrt(potenza_rumore), seg.shape)
    segmento_noisy = seg + rumore

    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(seg, label='ECG Pulito')
        plt.plot(segmento_noisy, label=f'ECG Rumoroso (SNR = {snr_db} dB)', alpha=0.7)
        plt.title("Simulazione Rumore su ECG")
        plt.xlabel("Campioni")
        plt.ylabel("Ampiezza")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    return segmento_noisy
