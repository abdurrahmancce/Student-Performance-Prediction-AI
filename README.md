# Student Performance Prediction AI

Predicts whether a student will **Pass** or **Fail** based on attendance
(0-10), class test marks (0-10), mid marks (0-30), previous semester GPA
(0-4.0), and weekly study hours (0-40) — using a properly trained and
evaluated machine learning pipeline.

## What changed from the original version

The original project generated `result` directly from a fixed formula
(`2*attendance + 3*ct + mid + 5*gpa >= 40`) and then trained a model on that
same formula. That means the model wasn't really *learning* anything — it
was just re-deriving arithmetic that was already known, which is why it hit
98% accuracy trivially and had no real predictive value. It was also
imbalanced (78% Pass / 22% Fail).

This version fixes that end-to-end:

| Area | Before | Now |
|---|---|---|
| Dataset | Deterministic formula, no noise | Correlated features from a latent "ability" factor + independent noise + 4% label noise, like real grading |
| Class balance | 78% / 22% | ~56% / 44% |
| Features | 4 | 5 (added study hours/week) |
| Preprocessing | None | `StandardScaler`, fit on train only (no leakage) |
| Model selection | Single Logistic Regression | 4 models compared (Logistic Regression, Random Forest, Gradient Boosting, SVM) via 5-fold cross-validation |
| Evaluation | Accuracy only | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report |
| Artifacts | None saved | Model + scaler saved with `joblib`, plus JSON metrics and PNG charts |
| Prediction | Hardcoded to run inline after training | Standalone `predict.py` with input validation and confidence score |

Realistic result: **~85% test accuracy**, evaluated on data the model never
trained on, with all four candidate models landing close together (a good
sign the problem is well-posed, not accidentally leaked or memorized).

## Project structure

```
student-performance-ai/
├── data/
│   └── dataset.csv              # generated dataset
├── models/
│   ├── best_model.joblib        # trained, selected model
│   ├── scaler.joblib            # fitted StandardScaler
│   └── model_meta.json          # which model won + its metrics
├── reports/
│   ├── evaluation_report.json   # metrics for all 4 candidate models
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── feature_importance.png
├── src/
│   ├── generate_dataset.py
│   ├── train_model.py
│   ├── predict.py
│   └── gui_app.py               # modern desktop GUI (Tkinter + ttkbootstrap)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Desktop GUI (recommended)

A modern desktop app (Tkinter + ttkbootstrap) ties everything together —
no terminal typing required after setup:

```bash
python src/gui_app.py
```

- **Predict tab** — sliders + precise spinboxes for each stat, a live
  Pass/Fail result card with a confidence bar, and a note showing which
  model is active and its accuracy.
- **Model Insights tab** — accuracy/precision/recall/F1/ROC-AUC as stat
  cards, plus the confusion matrix and model comparison charts, with a
  Refresh button.
- **Data & Training tab** — buttons to regenerate the dataset or retrain
  the model, with a live streaming log and progress indicator (runs in the
  background so the UI never freezes). Insights and Predict auto-update
  once training finishes.
- **Light/Dark mode** toggle in the sidebar.

If you'd rather run everything from the command line, the individual
scripts below still work exactly as before.

## Command-line usage

**1. Generate the dataset** (already included, but reproducible):
```bash
python src/generate_dataset.py
```

**2. Train & evaluate models:**
```bash
python src/train_model.py
```
This trains 4 models, cross-validates each, prints a full comparison,
picks the best one by cross-validation F1 score (not test-set luck), and
saves the model, scaler, and evaluation charts.

**3. Predict for a student:**
```bash
# Interactive
python src/predict.py

# Or pass values directly
python src/predict.py --attendance 8 --ct 8 --mid 24 --gpa 3.4 --study 12
```

## Possible next steps

- Swap in your real class data instead of the synthetic generator.
- Add a simple Streamlit UI so classmates/faculty can use it without a terminal.
- Track per-semester model drift if you keep collecting real results.
- Add SHAP values for per-student explanation ("why did the model predict Fail?").
