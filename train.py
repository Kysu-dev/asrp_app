"""
train.py
SVM classifier + GridSearchCV untuk tuning C & gamma otomatis
"""

import os
import sys
import pickle
import datetime
import numpy as np
from tqdm import tqdm
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve,
    GroupKFold,
    GroupShuffleSplit,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from preprocess import load_and_clean, extract_mfcc

# ── CONFIG ─────────────────────────────────────────────
FEATURES_PATH  = "models/features.pkl"
RUN_BASE_DIR   = os.path.join("models", "runs")
FINAL_MODEL_PATH  = os.path.join("models", "asr_model.pkl")
FINAL_SCALER_PATH = os.path.join("models", "scaler.pkl")
TEST_SIZE      = 0.25
AUGMENT_TRAIN  = True
AUGMENT_COPIES = 5   # dinaikkan: 5 versi aug per file train → lebih banyak variasi

# Grid parameter — diperluas agar hyperparameter tuning lebih optimal
PARAM_GRID = {
    "svc__C":     [0.1, 1, 10, 50, 100],
    "svc__gamma": ["scale", "auto", 0.001, 0.005, 0.01, 0.05],
    "svc__kernel":["rbf"]
}

# Parameter SVM base
SVM_BASE = {
    "probability":  True,
    "class_weight": "balanced"
}


# ── LOAD DATA ─────────────────────────────────────────
def load_features():
    with open(FEATURES_PATH, "rb") as f:
        data = pickle.load(f)
    return data["X"], data["y"], data["classes"], data.get("groups"), data.get("paths")


# ── RUN DIR ───────────────────────────────────────────
def make_run_dir():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RUN_BASE_DIR, ts)
    os.makedirs(path, exist_ok=True)
    return path


# ── CONFUSION MATRIX ──────────────────────────────────
def plot_confusion_matrix(cm, classes, run_dir):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=11)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (SVM + GridSearchCV)")
    fig.tight_layout()

    path = os.path.join(run_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Saved:", path)


# ── F1 / PREC / REC ───────────────────────────────────
def plot_f1_per_class(y_test, y_pred, classes, run_dir):
    f1   = f1_score(y_test, y_pred, average=None, zero_division=0)
    prec = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec  = recall_score(y_test, y_pred, average=None, zero_division=0)

    x = np.arange(len(classes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, prec, w, label="Precision", color="#4C72B0")
    ax.bar(x,     rec,  w, label="Recall",    color="#DD8452")
    ax.bar(x + w, f1,   w, label="F1",        color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.15)
    ax.set_title("Precision / Recall / F1 per Class (SVM + GridSearchCV)")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(run_dir, "f1_per_class.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Saved:", path)


# ── CV SCORE ──────────────────────────────────────────
def plot_cv_scores(cv_scores, run_dir):
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(cv_scores)), cv_scores * 100, color="#4C72B0", alpha=0.85)
    ax.axhline(cv_scores.mean() * 100, linestyle="--", color="red",
               label=f"Mean: {cv_scores.mean()*100:.1f}%")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cross Validation Scores (SVM + GridSearchCV)")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(run_dir, "cv_scores.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Saved:", path)


# ── LEARNING CURVE ───────────────────────────────────
def plot_learning_curve(train_sizes, train_scores, test_scores, run_dir):
    train_mean = train_scores.mean(axis=1) * 100
    train_std  = train_scores.std(axis=1)  * 100
    test_mean  = test_scores.mean(axis=1)  * 100
    test_std   = test_scores.std(axis=1)   * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_mean, marker="o", label="Train",  color="#4C72B0")
    ax.plot(train_sizes, test_mean,  marker="o", label="CV Val", color="#DD8452")

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="#4C72B0")
    ax.fill_between(train_sizes, test_mean  - test_std,  test_mean  + test_std,  alpha=0.2, color="#DD8452")

    ax.set_ylim(0, 110)
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Learning Curve (SVM + GridSearchCV)")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(run_dir, "learning_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Saved:", path)


# ── GRIDSEARCH HEATMAP ────────────────────────────────
def plot_gridsearch_heatmap(grid_result, run_dir):
    """Visualisasi skor GridSearchCV untuk setiap kombinasi C & gamma."""
    results = grid_result.cv_results_
    params  = results["params"]

    # Kumpulkan semua nilai unik C dan gamma
    Cs     = sorted(set(p["svc__C"]     for p in params))
    gammas = sorted(set(str(p["svc__gamma"]) for p in params))

    matrix = np.zeros((len(Cs), len(gammas)))
    for p, score in zip(params, results["mean_test_score"]):
        i = Cs.index(p["svc__C"])
        j = gammas.index(str(p["svc__gamma"]))
        matrix[i, j] = score * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    fig.colorbar(im, ax=ax, label="CV Accuracy (%)")

    ax.set_xticks(range(len(gammas)))
    ax.set_yticks(range(len(Cs)))
    ax.set_xticklabels(gammas, rotation=30, ha="right")
    ax.set_yticklabels([str(c) for c in Cs])
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    ax.set_title("GridSearchCV: CV Accuracy per (C, gamma)")

    for i in range(len(Cs)):
        for j in range(len(gammas)):
            ax.text(j, i, f"{matrix[i,j]:.1f}%", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    path = os.path.join(run_dir, "gridsearch_heatmap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("Saved:", path)


# ── MAIN TRAIN ────────────────────────────────────────
def train():
    print("=== TRAINING SVM ASR (GridSearchCV) ===\n")

    X, y, classes, groups, paths = load_features()
    y = np.array(y)

    # ── DEBUG ──
    print("\n=== FEATURE DEBUG ===")
    print(f"X shape     : {X.shape}")
    print(f"feature dim : {X.shape[1]}")
    print(f"class count : {len(classes)}")

    print("\nDistribusi kelas:")
    for c in classes:
        print(f"  {c}: {np.sum(y == int(c))}")

    run_dir = make_run_dir()
    print("\nRun dir:", run_dir)

    # ── SPLIT ─────────────────────────────────────────
    # PERBAIKAN: Gunakan GroupShuffleSplit agar speaker tidak bocor
    # antara train dan test set.
    # Dengan stratified split biasa, rekaman speaker A bisa ada di
    # train DAN test → model hafal suara → test acc 99% tapi CV 63%.
    # GroupShuffleSplit memastikan 1 speaker penuh masuk test set,
    # konsisten dengan GroupKFold yang dipakai di CV.
    indices = np.arange(len(X))
    if groups is not None and len(set(groups)) >= 2:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
        train_idx, test_idx = next(gss.split(indices, y, groups=groups))
        print(f"[INFO] Memakai GroupShuffleSplit — speaker di test set: "
              f"{set(groups[i] for i in test_idx)}")
    else:
        print("[WARN] Tidak ada info speaker, fallback ke StratifiedKFold split.")
        train_idx, test_idx = train_test_split(
            indices, test_size=TEST_SIZE, stratify=y, random_state=42
        )

    groups = np.array(groups) if groups is not None else None
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ── AUGMENT TRAINING DATA ─────────────────────────
    # aug_groups dibangun bersamaan agar panjangnya sama dengan X_train setelah augmentasi
    base_groups = np.array([groups[i] for i in train_idx]) if groups is not None else None
    aug_groups_list = []  # kumpulkan grup untuk sampel augmentasi

    if AUGMENT_TRAIN and paths is not None:
        aug_features, aug_labels = [], []
        total_aug = len(train_idx) * AUGMENT_COPIES
        print(f"\nAugmentasi data training ({total_aug} sampel)...", flush=True)
        with tqdm(total=total_aug, desc="Augment", unit="file", ncols=70) as pbar:
            for idx in train_idx:
                for _ in range(AUGMENT_COPIES):
                    try:
                        audio = load_and_clean(paths[idx], augment=True)
                        feat  = extract_mfcc(audio)
                        aug_features.append(feat)
                        aug_labels.append(y[idx])
                        if groups is not None:
                            aug_groups_list.append(groups[idx])  # sama dengan sampel aslinya
                    except Exception as e:
                        tqdm.write(f"[SKIP AUG] {paths[idx]}: {e}")
                    pbar.update(1)

        if aug_features:
            X_train = np.vstack([X_train, np.array(aug_features)])
            y_train = np.concatenate([y_train, np.array(aug_labels)])
            print(f"Augmented training size: {len(X_train)}", flush=True)

    # ── SCALING ───────────────────────────────────────
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    total_candidates = len(PARAM_GRID["svc__C"]) * len(PARAM_GRID["svc__gamma"]) * len(PARAM_GRID["svc__kernel"])
    print(f"\n=== GridSearchCV mulai: {total_candidates} kombinasi parameter ===", flush=True)
    print("(n_jobs=1 agar progress terlihat real-time di terminal)", flush=True)

    pipeline = Pipeline([("svc", SVC(**SVM_BASE))])

    # Pilih CV strategy untuk GridSearch.
    # Gunakan GroupKFold agar speaker tidak bocor antar fold CV.
    if groups is not None:
        # Gabungkan base_groups + aug_groups agar panjangnya sama dengan X_train
        if aug_groups_list:
            train_groups_arr = np.concatenate([base_groups, np.array(aug_groups_list)])
        else:
            train_groups_arr = base_groups
        n_groups = len(set(train_groups_arr))
        n_splits = min(n_groups, 5)
        if n_splits >= 2:
            print(f"[INFO] GridSearchCV memakai GroupKFold (n_splits={n_splits})", flush=True)
            inner_cv = GroupKFold(n_splits=n_splits)
        else:
            print("[WARN] Terlalu sedikit grup, fallback ke StratifiedKFold.", flush=True)
            inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            train_groups_arr = None
    else:
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_groups_arr = None

    grid_search = GridSearchCV(
        pipeline, PARAM_GRID,
        cv=inner_cv,
        scoring="accuracy",
        n_jobs=-1,   # pakai semua CPU untuk mempercepat
        verbose=1, refit=True
    )

    # Lewatkan groups hanya jika GroupKFold digunakan
    if train_groups_arr is not None:
        grid_search.fit(X_train_sc, y_train, groups=train_groups_arr)
    else:
        grid_search.fit(X_train_sc, y_train)

    best_params = grid_search.best_params_
    best_cv     = grid_search.best_score_
    print(f"\n[OK] Best params : {best_params}", flush=True)
    print(f"[OK] Best CV acc : {best_cv*100:.2f}%", flush=True)

    # ── EVALUATE BEST MODEL ───────────────────────────
    best_model = grid_search.best_estimator_.named_steps["svc"]
    y_pred = grid_search.predict(X_test_sc)

    train_acc = grid_search.score(X_train_sc, y_train)
    test_acc  = grid_search.score(X_test_sc,  y_test)

    report = classification_report(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)

    print("\n==============================")
    print(f"Train Accuracy : {train_acc*100:.2f}%")
    print(f"Test Accuracy  : {test_acc*100:.2f}%")
    print("==============================\n")
    print(report)

    # ── CV EVALUATION (best params) ───────────────────
    # PERBAIKAN: Scaler di-fit hanya di dalam tiap fold (via Pipeline),
    # bukan di-fit pada seluruh X sebelum CV → mencegah data leakage minor.
    cv_pipeline = make_pipeline(StandardScaler(), SVC(**SVM_BASE, **{
        "C":      best_params["svc__C"],
        "gamma":  best_params["svc__gamma"],
        "kernel": best_params["svc__kernel"]
    }))

    # n_splits_cv diinisialisasi dulu agar tidak NameError di bagian learning curve
    n_splits_cv = 0
    if groups is not None:
        n_groups_all = len(set(groups))
        n_splits_cv  = min(5, n_groups_all)
        if n_splits_cv < 2:
            outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(cv_pipeline, X, y, cv=outer_cv)
        else:
            gkf = GroupKFold(n_splits=n_splits_cv)
            cv_scores = cross_val_score(cv_pipeline, X, y, cv=gkf, groups=groups)
    else:
        outer_cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(cv_pipeline, X, y, cv=outer_cv)

    # ── LEARNING CURVE ────────────────────────────────
    lc_model = make_pipeline(StandardScaler(), SVC(**SVM_BASE, **{
        "C":      best_params["svc__C"],
        "gamma":  best_params["svc__gamma"],
        "kernel": best_params["svc__kernel"]
    }))

    if groups is not None and n_splits_cv >= 2:
        lc_cv = GroupKFold(n_splits=n_splits_cv)
    else:
        lc_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(
        lc_model, X, y,
        cv=lc_cv,
        groups=groups if (groups is not None and n_splits_cv >= 2) else None,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="accuracy",
        shuffle=True,
        random_state=42
    )

    print("\nCross Validation (best params):")
    print(f"  Mean : {cv_scores.mean()*100:.2f}%")
    print(f"  Std  : {cv_scores.std()*100:.2f}%")

    # ── SAVE MODEL ────────────────────────────────────
    model_path  = os.path.join(run_dir, "svm_model.pkl")
    scaler_path = os.path.join(run_dir, "scaler.pkl")

    os.makedirs("models", exist_ok=True)

    with open(model_path,  "wb") as f: pickle.dump(best_model, f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
    with open(FINAL_MODEL_PATH,  "wb") as f: pickle.dump(best_model, f)
    with open(FINAL_SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    # ── SAVE REPORT ───────────────────────────────────
    report_path = os.path.join(run_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=== SVM TRAINING REPORT (GridSearchCV) ===\n\n")

        f.write("=== FEATURE DEBUG ===\n")
        f.write(f"X shape     : {X.shape}\n")
        f.write(f"feature dim : {X.shape[1]}\n\n")

        f.write("=== GRIDSEARCH RESULT ===\n")
        f.write(f"Best params : {best_params}\n")
        f.write(f"Best CV acc : {best_cv*100:.2f}%\n\n")

        f.write(f"Train Accuracy : {train_acc*100:.2f}%\n")
        f.write(f"Test Accuracy  : {test_acc*100:.2f}%\n\n")
        f.write(f"CV Mean        : {cv_scores.mean()*100:.2f}%\n")
        f.write(f"CV Std         : {cv_scores.std()*100:.2f}%\n\n")

        f.write("Class Distribution:\n")
        for c in classes:
            f.write(f"  {c}: {np.sum(y == int(c))}\n")

        f.write("\nCLASSIFICATION REPORT:\n")
        f.write(report)

    # ── PLOTS ─────────────────────────────────────────
    plot_confusion_matrix(cm, classes, run_dir)
    plot_f1_per_class(y_test, y_pred, classes, run_dir)
    plot_cv_scores(cv_scores, run_dir)
    plot_learning_curve(train_sizes, train_scores, test_scores, run_dir)
    plot_gridsearch_heatmap(grid_search, run_dir)

    print(f"\nDONE -> {run_dir}")
    print(f"\n[BEST] Best Params  : C={best_params['svc__C']}, gamma={best_params['svc__gamma']}")
    print(f"[BEST] Test Accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    train()