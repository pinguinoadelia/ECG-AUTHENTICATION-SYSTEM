import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

def svm_pipeline(
    X_train, X_test, y_train, y_test,
    plots_dir="plots"
):
    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    print("Modello SVM addestrato correttamente.")

    y_pred = svm_model.predict(X_test)
    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average='macro')
    prec = acc
    print(f"Accuracy: {acc:.4f}")
    print(f"Precisione (macro): {prec:.4f}")
    print("Matrice di Confusione SVM:")
    print(confusion_matrix(y_test, y_pred))

    cm_svm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm_svm, interpolation='nearest', cmap='Blues')
    plt.title("Matrice di Confusione - SVM")
    plt.colorbar()
    ticks = np.arange(len(np.unique(y_test)))
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i in range(cm_svm.shape[0]):
        for j in range(cm_svm.shape[1]):
            plt.text(j, i, f"{cm_svm[i, j]}", ha='center', va='center')
    plt.tight_layout()
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f"{plots_dir}/confusion_matrix_svm.png")
    plt.show()

    far_list, frr_list = [], []
    for lab in np.unique(y_test):
        tp = np.sum((y_pred == lab) & (y_test == lab))
        fn = np.sum((y_pred != lab) & (y_test == lab))
        fp = np.sum((y_pred == lab) & (y_test != lab))
        pos = np.sum(y_test == lab)
        neg = np.sum(y_test != lab)
        frr_list.append(fn/pos if pos else 0)
        far_list.append(fp/neg if neg else 0)

    mean_frr = np.mean(frr_list)
    mean_far = np.mean(far_list)
    eer      = (mean_frr + mean_far) / 2
    print(f"FRR: {mean_frr:.4f}, FAR: {mean_far:.4f}, EER: {eer:.4f}")

    plt.figure(figsize=(8,5))
    labels   = ['Accuracy', 'Precision', 'FAR', 'FRR', 'EER']
    svm_vals = [acc, prec, mean_far, mean_frr, eer]
    colors   = ['green', 'blue', 'red', 'orange', 'purple']
    plt.bar(labels, svm_vals, color=colors)
    plt.ylim(0,1)
    plt.title("Metriche di performance - SVM")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    return svm_model, y_pred, {
        'acc': acc, 'prec': prec,
        'far': mean_far, 'frr': mean_frr, 'eer': eer
    }
