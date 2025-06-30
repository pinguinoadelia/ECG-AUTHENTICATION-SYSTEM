import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

def cnn_lstm_pipeline(
    X_train_dl, X_test_dl, y_train_dl, y_test_dl,
    plots_dir="plots"
):
    y_tr_cat = to_categorical(y_train_dl)
    y_te_cat = to_categorical(y_test_dl)

    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=(X_train_dl.shape[1],1)),
        MaxPooling1D(pool_size=2), Dropout(0.3),
        Conv1D(64, kernel_size=3, activation='relu'), MaxPooling1D(pool_size=2), Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dense(64, activation='relu'), Dropout(0.3),
        Dense(y_te_cat.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train_dl, y_tr_cat,
        epochs=30,
        batch_size=32,
        validation_data=(X_test_dl, y_te_cat),
        callbacks=[es],
        verbose=1
    )

    y_pred_dl = np.argmax(model.predict(X_test_dl), axis=1)
    acc_dl    = accuracy_score(y_test_dl, y_pred_dl)
    prec_dl   = precision_score(y_test_dl, y_pred_dl, average='macro')
    prec_dl   = acc_dl
    print(f"[DL] Accuracy: {acc_dl:.4f}")
    print(f"[DL] Precisione (macro): {prec_dl:.4f}")

    cm_dl = confusion_matrix(y_test_dl, y_pred_dl)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm_dl, interpolation='nearest', cmap='Blues')
    plt.title("Matrice di Confusione - CNN-LSTM")
    plt.colorbar()
    ticks = np.arange(len(np.unique(y_test_dl)))
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i in range(cm_dl.shape[0]):
        for j in range(cm_dl.shape[1]):
            plt.text(j, i, f"{cm_dl[i, j]}", ha='center', va='center')
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f"{plots_dir}/confusion_matrix_dl.png")
    plt.show()

    far_dl, frr_dl = [], []
    for lab in np.unique(y_test_dl):
        tp = np.sum((y_pred_dl == lab) & (y_test_dl == lab))
        fn = np.sum((y_pred_dl != lab) & (y_test_dl == lab))
        fp = np.sum((y_pred_dl == lab) & (y_test_dl != lab))
        pos = np.sum(y_test_dl == lab)
        neg = np.sum(y_test_dl != lab)
        frr_dl.append(fn/pos if pos else 0)
        far_dl.append(fp/neg if neg else 0)

    mean_frr_dl = np.mean(frr_dl)
    mean_far_dl = np.mean(far_dl)
    eer_dl      = (mean_frr_dl + mean_far_dl) / 2
    print(f"[DL] FRR: {mean_frr_dl:.4f}, FAR: {mean_far_dl:.4f}, EER: {eer_dl:.4f}")

    plt.figure(figsize=(8,5))
    labels = ['Accuracy', 'Precision', 'FAR', 'FRR', 'EER']
    dl_vals = [acc_dl, prec_dl, mean_far_dl, mean_frr_dl, eer_dl]
    colors  = ['green', 'blue', 'red', 'orange', 'purple']
    plt.bar(labels, dl_vals, color=colors)
    plt.ylim(0,1)
    plt.title("Metriche di performance - CNN-LSTM")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    return model, y_pred_dl, {
        'acc': acc_dl, 'prec': prec_dl,
        'far': mean_far_dl, 'frr': mean_frr_dl, 'eer': eer_dl
    }
