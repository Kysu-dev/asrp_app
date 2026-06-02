"""
train.py
SVM classifier + DEBUG FEATURE CONSISTENCY
"""

import os
import pickle
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ── CONFIG ─────────────────────────────────────────────
FEATURES_PATH = "models/features.pkl"
RUN_BASE_DIR = os.path.join("models", "runs")
FINAL_MODEL_PATH = os.path.join("models", "asr_model.pkl")
FINAL_SCALER_PATH = os.path.join("models", "scaler.pkl")
TEST_SIZE = 0.2

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 10,
    "gamma": "scale",
    "probability": True,
    "class_weight": "balanced"
}


# ── LOAD DATA ─────────────────────────────────────────
def load_features():
    with open(FEATURES_PATH, "rb") as f:
        data = pickle.load(f)
    return data["X"], data["y"], data["classes"]


# ── RUN DIR ───────────────────────────────────────────
def make_run_dir():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RUN_BASE_DIR, ts)
    os.makedirs(path, exist_ok=True)
    return path


# ── CONFUSION MATRIX ──────────────────────────────────
def plot_confusion_matrix(cm, classes, run_dir):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")

    plt.xticks(range(len(classes)), classes, rotation=45)
    plt.yticks(range(len(classes)), classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.title("Confusion Matrix (SVM)")
    plt.colorbar()

    path = os.path.join(run_dir, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print("Saved:", path)


# ── F1 / PREC / REC ───────────────────────────────────
def plot_f1_per_class(y_test, y_pred, classes, run_dir):
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    prec = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec = recall_score(y_test, y_pred, average=None, zero_division=0)

    x = np.arange(len(classes))
    w = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - w, prec, w, label="Precision")
    plt.bar(x, rec, w, label="Recall")
    plt.bar(x + w, f1, w, label="F1")

    plt.xticks(x, classes)
    plt.ylim(0, 1.1)
    plt.title("Precision / Recall / F1 (SVM)")
    plt.legend()

    path = os.path.join(run_dir, "f1_per_class.png")
    plt.savefig(path)
    plt.close()
    print("Saved:", path)


# ── CV SCORE ──────────────────────────────────────────
def plot_cv_scores(cv_scores, run_dir):
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(cv_scores)), cv_scores * 100)
    plt.axhline(cv_scores.mean() * 100, linestyle="--")
    plt.ylim(0, 100)
    plt.title("Cross Validation Scores (SVM)")

    path = os.path.join(run_dir, "cv_scores.png")
    plt.savefig(path)
    plt.close()
    print("Saved:", path)


# ── MAIN TRAIN ────────────────────────────────────────
def train():
    print("=== TRAINING SVM ASR ===\n")

    X, y, classes = load_features()
    y = np.array(y)

    # 🔥 DEBUG INPUT SHAPE
    print("\n=== FEATURE DEBUG ===")
    print(f"X shape           : {X.shape}")
    print(f"y shape           : {y.shape}")
    print(f"feature dim       : {X.shape[1]}")
    print(f"class count       : {len(classes)}")

    print("\nDistribusi kelas:")
    for c in classes:
        print(f"{c}: {np.sum(y == int(c))}")

    run_dir = make_run_dir()
    print("\nRun dir:", run_dir)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=42
    )

    # scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # model
    model = SVC(**SVM_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n==============================")
    print(f"Train Accuracy : {train_acc*100:.2f}%")
    print(f"Test Accuracy  : {test_acc*100:.2f}%")
    print("==============================\n")
    print(report)

    # CV
    scaler_cv = StandardScaler()
    X_cv = scaler_cv.fit_transform(X)

    cv_scores = cross_val_score(
        SVC(**SVM_PARAMS),
        X_cv,
        y,
        cv=5
    )

    print("\nCross Validation:")
    print(f"Mean : {cv_scores.mean()*100:.2f}%")
    print(f"Std  : {cv_scores.std()*100:.2f}%")

    # SAVE MODEL
    model_path = os.path.join(run_dir, "svm_model.pkl")
    scaler_path = os.path.join(run_dir, "scaler.pkl")

    os.makedirs("models", exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    with open(FINAL_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(FINAL_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # SAVE REPORT (FULL DEBUG)
    report_path = os.path.join(run_dir, "classification_report.txt")

    with open(report_path, "w") as f:
        f.write("=== SVM TRAINING REPORT ===\n\n")

        f.write("=== FEATURE DEBUG ===\n")
        f.write(f"X shape : {X.shape}\n")
        f.write(f"feature dim : {X.shape[1]}\n\n")

        f.write(f"Train Accuracy : {train_acc*100:.2f}%\n")
        f.write(f"Test Accuracy  : {test_acc*100:.2f}%\n\n")
        f.write(f"CV Mean        : {cv_scores.mean()*100:.2f}%\n")
        f.write(f"CV Std         : {cv_scores.std()*100:.2f}%\n\n")

        f.write("Class Distribution:\n")
        for c in classes:
            f.write(f"{c}: {np.sum(y == int(c))}\n")

        f.write("\nCLASSIFICATION REPORT:\n")
        f.write(report)

    # PLOTS (TETAP ADA)
    plot_confusion_matrix(cm, classes, run_dir)
    plot_f1_per_class(y_test, y_pred, classes, run_dir)
    plot_cv_scores(cv_scores, run_dir)

    print("\nDONE →", run_dir)


if __name__ == "__main__":
    train()