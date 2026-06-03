import os
import pickle
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

FEATURES_PATH    = "models/features.pkl"
RUN_BASE_DIR     = os.path.join("models", "runs")
FINAL_MODEL_PATH = os.path.join("models", "asr_model.pkl")
FINAL_SCALER_PATH= os.path.join("models", "scaler.pkl")
TEST_SIZE        = 0.2

MLP_PARAMS = {
    "hidden_layer_sizes": (256, 128),
    "activation"        : "relu",
    "solver"            : "adam",
    "alpha"             : 0.001,
    "max_iter"          : 500,
    "early_stopping"    : True,
    "random_state"      : 42
}

CV_FOLDS  = 3
N_JOBS_CV = -1


def load_features():
    with open(FEATURES_PATH, "rb") as f:
        data = pickle.load(f)
    return data["X"], data["y"], data.get("groups", np.arange(len(data["y"]))), data["classes"]


def make_run_dir():
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RUN_BASE_DIR, ts)
    os.makedirs(path, exist_ok=True)
    return path


def plot_confusion_matrix(cm, classes, run_dir):
    fig, ax = plt.subplots(figsize=(9, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix (MLP)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(run_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_metrics_per_class(y_test, y_pred, classes, run_dir):
    f1   = f1_score(y_test, y_pred, average=None, zero_division=0)
    prec = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_test, y_pred, average=None, zero_division=0)

    x = np.arange(len(classes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, prec, w, label="Precision", color="#4C72B0")
    ax.bar(x,     rec,  w, label="Recall",    color="#55A868")
    ax.bar(x + w, f1,   w, label="F1-Score",  color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.1)
    ax.set_title("Precision / Recall / F1 per Kelas (MLP)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.axhline(y=0.8, linestyle="--", color="gray", alpha=0.5, label="Target 80%")
    plt.tight_layout()

    path = os.path.join(run_dir, "metrics_per_class.png")
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cv_scores(cv_result, run_dir):
    train_scores = cv_result["train_score"] * 100
    test_scores  = cv_result["test_score"]  * 100

    folds = range(1, len(test_scores) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(folds, train_scores, "o-", label="Train",  color="#4C72B0")
    ax.plot(folds, test_scores,  "s-", label="Val",    color="#C44E52")
    ax.axhline(test_scores.mean(), linestyle="--", color="#C44E52", alpha=0.6,
               label=f"Val Mean: {test_scores.mean():.1f}%")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(list(folds))
    ax.set_title(f"Cross-Validation Scores ({CV_FOLDS}-Fold)", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(run_dir, "cv_scores.png")
    plt.savefig(path, dpi=150)
    plt.close()


def train():
    X, y, groups, classes = load_features()
    y = np.array(y)
    groups = np.array(groups)
    run_dir = make_run_dir()

    # Mencegah Data Leakage: Pastikan satu original file + semua augmentasinya berada di split yang sama
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = MLPClassifier(**MLP_PARAMS)
    model.fit(X_train, y_train)

    y_pred     = model.predict(X_test)
    train_acc  = model.score(X_train, y_train)
    test_acc   = model.score(X_test,  y_test)
    report     = classification_report(y_test, y_pred)
    cm         = confusion_matrix(y_test, y_pred)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPClassifier(**MLP_PARAMS))
    ])

    cv_result = cross_validate(
        pipe, X, y, groups=groups,
        cv=GroupKFold(n_splits=CV_FOLDS),
        scoring="accuracy",
        return_train_score=True,
        n_jobs=N_JOBS_CV
    )

    cv_test  = cv_result["test_score"]
    cv_train = cv_result["train_score"]

    model_path  = os.path.join(run_dir, "mlp_model.pkl")
    scaler_path = os.path.join(run_dir, "scaler.pkl")

    with open(model_path,        "wb") as f: pickle.dump(model,  f)
    with open(scaler_path,       "wb") as f: pickle.dump(scaler, f)
    with open(FINAL_MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(FINAL_SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    report_path = os.path.join(run_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Train Accuracy : {train_acc*100:.2f}%\n")
        f.write(f"Test  Accuracy : {test_acc*100:.2f}%\n")
        f.write(f"CV Val  Mean   : {cv_test.mean()*100:.2f}%\n")
        f.write(f"CV Train Mean  : {cv_train.mean()*100:.2f}%\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(report)

    plot_confusion_matrix(cm, classes, run_dir)
    plot_metrics_per_class(y_test, y_pred, classes, run_dir)
    plot_cv_scores(cv_result, run_dir)
    print(f"Finished. Saved to {run_dir}")


if __name__ == "__main__":
    train()