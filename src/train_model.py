import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

FEATURES = [
    "attendance", "ct_marks", "mid_marks",
    "previous_semester_gpa", "study_hours_per_week",
]
TARGET = "result"
RANDOM_STATE = 42

# Resolve paths relative to the project root, so this works regardless of
# the terminal's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_data(path=None):
    df = pd.read_csv(path or DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def build_candidate_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }


def evaluate_model(name, model, X_train, X_test, y_train, y_test, cv):
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "cv_f1_mean": float(np.mean(cv_scores)),
        "cv_f1_std": float(np.std(cv_scores)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred)),
        "test_recall": float(recall_score(y_test, y_pred)),
        "test_f1": float(f1_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
    }
    return model, metrics, y_pred, y_proba


def plot_confusion_matrix(cm, model_name, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fail", "Pass"]); ax.set_yticklabels(["Fail", "Pass"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results, out_path):
    names = list(results.keys())
    f1s = [results[n]["test_f1"] for n in names]
    cvs = [results[n]["cv_f1_mean"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width/2, cvs, width, label="CV F1 (train)")
    ax.bar(x + width/2, f1s, width, label="Test F1")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("F1 score")
    ax.set_title("Model comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names, out_path):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(importances)), importances[order])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in order], rotation=30, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Feature importance (best model)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURES, index=X_test.index)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models = build_candidate_models()
    results = {}
    fitted_models = {}
    preds = {}

    print("=" * 60)
    print(" Training & evaluating candidate models (5-fold CV + holdout test)")
    print("=" * 60)

    for name, model in models.items():
        fitted, metrics, y_pred, y_proba = evaluate_model(
            name, model, X_train_scaled, X_test_scaled, y_train, y_test, cv
        )
        results[name] = metrics
        fitted_models[name] = fitted
        preds[name] = y_pred

        print(f"\n{name}")
        print(f"  CV F1 (5-fold, train only): {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")
        print(f"  Test Accuracy : {metrics['test_accuracy']:.3f}")
        print(f"  Test Precision: {metrics['test_precision']:.3f}")
        print(f"  Test Recall   : {metrics['test_recall']:.3f}")
        print(f"  Test F1       : {metrics['test_f1']:.3f}")
        if metrics["test_roc_auc"] is not None:
            print(f"  Test ROC-AUC  : {metrics['test_roc_auc']:.3f}")

    # Pick best model by CV F1 (train-only signal, avoids picking a model that
    # just got lucky on the test split)
    best_name = max(results, key=lambda n: results[n]["cv_f1_mean"])
    best_model = fitted_models[best_name]
    best_pred = preds[best_name]

    print("\n" + "=" * 60)
    print(f" Best model selected: {best_name}")
    print("=" * 60)
    print(classification_report(y_test, best_pred, target_names=["Fail", "Pass"]))

    cm = confusion_matrix(y_test, best_pred)

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump({
            "model_name": best_name,
            "features": FEATURES,
            "metrics": results[best_name],
        }, f, indent=2)

    with open(REPORTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_confusion_matrix(cm, best_name, REPORTS_DIR / "confusion_matrix.png")
    plot_model_comparison(results, REPORTS_DIR / "model_comparison.png")
    plot_feature_importance(best_model, FEATURES, REPORTS_DIR / "feature_importance.png")

    print("\nSaved:")
    print("  models/best_model.joblib")
    print("  models/scaler.joblib")
    print("  models/model_meta.json")
    print("  reports/evaluation_report.json")
    print("  reports/confusion_matrix.png")
    print("  reports/model_comparison.png")
    if hasattr(best_model, "feature_importances_"):
        print("  reports/feature_importance.png")


if __name__ == "__main__":
    main()
