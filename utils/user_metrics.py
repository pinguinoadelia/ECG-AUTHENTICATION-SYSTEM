import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def save_user_metrics(
    y_test, y_pred, y_test_dl, y_pred_dl,
    health_labels, id_to_record, records_info,
    output_path="results/user_metrics.csv"
):
    """
    Genera il CSV con le sole colonne User_ID, Health, Aut_SVM e Aut_CNN-LSTM,
    e lo salva in `output_path`.
    """
    def is_auth(acc, prec, th_acc=0.85, th_prec=0.85):
        return "Successo" if acc >= th_acc and prec >= th_prec else "Insuccesso"

    rows = []
    for uid in np.unique(y_test):
        rec    = id_to_record[uid]
        health = records_info[rec]

        # SVM
        idx_s  = np.where(y_test == uid)[0]
        a_s    = accuracy_score(y_test[idx_s], y_pred[idx_s])
        p_s    = a_s
        auth_s = is_auth(a_s, p_s)

        # DL
        idx_d  = np.where(y_test_dl == uid)[0]
        a_d    = accuracy_score(y_test_dl[idx_d], y_pred_dl[idx_d])
        p_d    = a_d
        auth_d = is_auth(a_d, p_d)

        rows.append({
            "User_ID":       f"user_{uid}",
            "Health":        health,
            "Aut_SVM":       auth_s,
            "Aut_CNN-LSTM":  auth_d
        })

    # Costruisco e filtro le colonne
    df_users = pd.DataFrame(rows)[["User_ID", "Health", "Aut_SVM", "Aut_CNN-LSTM"]]

    # Salvo e stampo
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_users.to_csv(output_path, index=False)
    print(f"🗂️ Tabella per utente salvata in {output_path}")
    print(df_users)
