# 🎓 Student Performance Prediction AI

> **An end-to-end machine learning system that predicts whether a student is likely to _Pass_ or _Fail_ using academic performance, attendance, GPA, and study habits.**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20%26%20Evaluation-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-Educational-lightgrey?style=flat-square)](#-license)

---

## 📌 Overview

**Student Performance Prediction AI** is a machine learning project designed to estimate a student's academic outcome as **Pass** or **Fail**.

The system uses five input features:

- 📅 **Attendance:** 0–10
- 📝 **CT Marks:** 0–10
- 📚 **Mid Marks:** 0–30
- 🎓 **Previous Semester GPA:** 0–4.0
- ⏱️ **Study Hours per Week:** 0–40

The project goes beyond a simple classification script. It includes **synthetic data generation, preprocessing, model comparison, cross-validation, holdout evaluation, model persistence, command-line prediction, and a modern desktop GUI**.

> ⚠️ **Important:** The current dataset is synthetic. The model should be retrained and validated with real institutional data before being used for real academic decisions.

---

## 🎯 Project Goals

The main goals are to:

- 🔍 Identify students who may be academically at risk.
- 📊 Use multiple academic and behavioral indicators instead of a single score.
- 🤖 Compare several classification algorithms fairly.
- 🧪 Evaluate the selected model on unseen test data.
- 🖥️ Provide an easy-to-use desktop interface.
- 💾 Save trained model artifacts for future predictions.
- 📈 Generate evaluation reports and visualizations.

Early prediction can support timely academic intervention, targeted mentoring, and evidence-based student support.

---

## ✨ Key Features

### 🤖 Machine Learning Pipeline

- Realistic synthetic dataset with **1,200 simulated students**
- Five predictive features
- Correlated features based on a latent student ability/discipline factor
- Independent feature noise
- **4% label noise** to simulate real-world grading inconsistency
- Approximately **57% Pass / 43% Fail** class distribution
- Stratified **80/20 train-test split**
- `StandardScaler` fitted only on training data
- **5-fold stratified cross-validation**
- Four candidate classification models
- Model selection based on **cross-validation F1 score**
- Multiple evaluation metrics

### 🧠 Models Compared

| Model | Purpose |
|---|---|
| Logistic Regression | Strong linear baseline |
| Random Forest | Non-linear ensemble learning |
| Gradient Boosting | Sequential ensemble learning |
| SVM (RBF) | Non-linear decision boundaries |

### 📊 Evaluation

The project evaluates:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report
- Cross-validation F1

The current project presentation reports approximately:

| Metric | Test Result |
|---|---:|
| 🎯 Accuracy | **85.0%** |
| 🎯 Precision | **86.2%** |
| 🎯 Recall | **87.5%** |
| 🎯 F1 Score | **86.9%** |
| 🎯 ROC-AUC | **92.1%** |

These results were measured on **240 students not used for training**.

---

## 🧩 Why This Version Is More Reliable

The original approach generated the target label directly from a fixed mathematical formula and then trained a model to reproduce that formula. This produced very high accuracy but did not represent meaningful machine learning.

The improved pipeline addresses that problem by introducing:

- 🧬 A latent ability/discipline factor
- 🎲 Independent feature noise
- 🎲 Unexplained score variation
- 🔀 4% intentional label noise
- ⚖️ More balanced Pass/Fail classes
- 🧪 Proper train/test separation
- 📏 Training-only feature scaling
- 🔎 Cross-validation-based model selection

This makes the experiment more representative of a real classification problem.

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   Student Dataset    │
                    │  1,200 observations   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Preparation     │
                    │ • Feature selection  │
                    │ • Train/Test split   │
                    │ • StandardScaler     │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │       Model Comparison          │
              │                                 │
              │ Logistic Regression             │
              │ Random Forest                   │
              │ Gradient Boosting               │
              │ SVM (RBF)                       │
              └───────────────┬─────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │  5-Fold CV + Test    │
                    │     Evaluation       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Best Model by      │
                    │   CV F1 Score        │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
       ┌──────────────────┐        ┌──────────────────┐
       │ CLI Prediction   │        │ Desktop GUI      │
       │ predict.py       │        │ Tkinter +        │
       │                  │        │ ttkbootstrap     │
       └──────────────────┘        └──────────────────┘
```

---

## 📂 Project Structure

```text
Student-Performance-Prediction-AI/
│
├── 📁 data/
│   └── dataset.csv
│
├── 📁 models/
│   ├── best_model.joblib
│   ├── scaler.joblib
│   └── model_meta.json
│
├── 📁 reports/
│   ├── evaluation_report.json
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
├── 📁 src/
│   ├── generate_dataset.py
│   ├── train_model.py
│   ├── predict.py
│   └── gui_app.py
│
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🗃️ Dataset

The dataset is generated programmatically, making the experiment reproducible.

### Input Features

| Feature | Range | Description |
|---|---:|---|
| `attendance` | 0–10 | Attendance score |
| `ct_marks` | 0–10 | Class test marks |
| `mid_marks` | 0–30 | Midterm examination marks |
| `previous_semester_gpa` | 0–4.0 | Previous semester GPA |
| `study_hours_per_week` | 0–40 | Weekly study time |
| `result` | 0/1 | Target: Fail / Pass |

The generator uses a fixed random seed and creates **1,200 samples**. The target is generated from a weighted performance score with additional random variation, followed by intentional label noise.

---

## 🛠️ Tech Stack

### Programming & Data

- 🐍 Python
- 🐼 Pandas
- 🔢 NumPy

### Machine Learning

- 🤖 scikit-learn
- 📏 StandardScaler
- 🌲 Random Forest
- 📈 Gradient Boosting
- 📐 Logistic Regression
- 🔷 SVM

### Visualization

- 📊 Matplotlib

### Application

- 🖥️ Tkinter
- 🎨 ttkbootstrap
- 🖼️ Pillow

### Model Persistence

- 💾 Joblib

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/abdurrahmancce/Student-Performance-Prediction-AI.git
cd Student-Performance-Prediction-AI
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Required packages include:

```text
numpy
pandas
scikit-learn
matplotlib
joblib
ttkbootstrap
pillow
```

---

## 🚀 Usage

### Step 1: Generate Dataset

```bash
python src/generate_dataset.py
```

This creates:

```text
data/dataset.csv
```

---

### Step 2: Train and Evaluate Models

```bash
python src/train_model.py
```

The training pipeline:

1. Loads the dataset.
2. Separates features and target.
3. Creates a stratified 80/20 train-test split.
4. Fits `StandardScaler` using training data only.
5. Runs 5-fold cross-validation.
6. Trains four candidate models.
7. Evaluates each model on unseen test data.
8. Selects the best model using CV F1.
9. Saves the trained model and scaler.
10. Generates evaluation reports and charts.

---

### Step 3: Run Prediction

#### Interactive mode

```bash
python src/predict.py
```

Enter:

```text
Attendance
CT Marks
Mid Marks
Previous Semester GPA
Study Hours per Week
```

The system returns:

```text
Prediction: PASS ✅
Confidence: XX.X%
```

or:

```text
Prediction: FAIL ❌
Confidence: XX.X%
```

#### Direct command-line prediction

```bash
python src/predict.py --attendance 8 --ct 8 --mid 24 --gpa 3.4 --study 12
```

---

## 🖥️ Desktop GUI

The recommended way to use the project is the desktop application.

```bash
python src/gui_app.py
```

### 🎯 Predict Tab

- Input sliders
- Precise spinboxes
- Pass/Fail result card
- Confidence indicator
- Active model information
- Test accuracy display

### 📊 Model Insights Tab

Displays:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Model Comparison
- Feature Importance

### ⚙️ Data & Training Tab

Allows users to:

- Generate the dataset
- Retrain the model
- View live training logs
- Monitor training progress

### 🌙 Theme Support

The application also provides:

- ☀️ Light Mode
- 🌙 Dark Mode

---

## 📈 Evaluation Workflow

The training process is designed to reduce leakage and unreliable model selection.

```text
Raw Dataset
    │
    ▼
Stratified 80/20 Split
    │
    ├──────────────► Test Set
    │                 20%
    │
    ▼
Training Set
    │
    ▼
StandardScaler
    │
    ▼
5-Fold Cross Validation
    │
    ▼
Compare 4 Models
    │
    ▼
Select Best CV F1
    │
    ▼
Evaluate Once on Test Set
    │
    ▼
Save Model + Metrics + Charts
```

---

## 📊 Generated Reports

After training, the project creates:

### `evaluation_report.json`

Contains evaluation metrics for all candidate models.

### `confusion_matrix.png`

Shows:

- True Positives
- True Negatives
- False Positives
- False Negatives

### `model_comparison.png`

Compares:

- Cross-validation F1
- Test F1

### `feature_importance.png`

Generated when the selected model provides feature importance values.

---

## 🔍 Example Interpretation

The reported test confusion matrix contains:

- ✅ **119** correctly predicted Pass
- ✅ **85** correctly predicted Fail
- ⚠️ **19** false positives
- ⚠️ **17** false negatives

Recall is slightly higher than precision, which is useful for an early-warning scenario because the system is somewhat more willing to flag potential academic risk.

---

## 🎓 Potential Applications

This system can be adapted for:

- 🏫 University academic monitoring
- 👨‍🏫 Faculty advising
- 📚 Early academic intervention
- 🎯 Student mentoring
- 📊 Academic analytics
- 🔎 At-risk student identification
- 🧪 Machine learning education and experimentation

> The prediction should be treated as a **decision-support signal**, not as a final judgment about a student.

---

## 🔐 Responsible Use

Academic prediction systems can affect real students, so responsible deployment matters.

Before real-world use:

- Use validated institutional data.
- Check for demographic and institutional bias.
- Monitor false positives and false negatives.
- Protect student privacy.
- Obtain appropriate authorization for data use.
- Retrain and validate the model regularly.
- Keep human educators involved in decisions.

---

## 🧪 Limitations

Current limitations include:

- The dataset is synthetic.
- Real student behavior can be more complex.
- The model predicts only Pass/Fail.
- No longitudinal semester data is currently used.
- No explainability layer such as SHAP is implemented yet.
- Model performance may change on real institutional data.

Therefore, the reported metrics should **not** be interpreted as expected real-world performance.

---

## 🚀 Future Improvements

### 🔹 Real Academic Dataset

Replace the synthetic generator with anonymized semester records.

### 🔹 Explainable AI

Add SHAP-based explanations such as:

> "The prediction was influenced most by attendance, GPA, and midterm performance."

### 🔹 Model Monitoring

Track model drift and performance across semesters.

### 🔹 Web Application

Build a Streamlit or Django-based web interface.

### 🔹 Model Deployment

Package the application with PyInstaller so users can run it without manually installing Python.

### 🔹 Multi-Class Prediction

Extend the system from:

```text
Pass / Fail
```

to:

```text
At Risk / Average / High Performer
```

---

## 📸 Screenshots / Preview

>### 🎯 Prediction Interface

<img src="https://github.com/user-attachments/assets/409962a6-b6a2-4c81-9fe6-e6fe8caa4ea5" width="500">

>### 📊 Model Insights

<img src="https://github.com/user-attachments/assets/0c85dbef-da09-47de-880a-e275758c726c" width="500">

>### ⚙️ Data & Training

<img src="https://github.com/user-attachments/assets/69b16401-9f45-42e6-8ad3-87ac1f2c1bed" width="500">

>### 🌙 Dark Mode

<img src="https://github.com/user-attachments/assets/cec1e6fa-95d1-4cb7-9c56-0bdf055ce740" width="500">


---

## 👨‍💻 Author

>### **Abdur Rahman**

**Computer & Communication Engineering (CCE)**  
**International Islamic University Chittagong (IIUC)**

🔗 GitHub: **https://github.com/abdurrahmancce**

🔗 Repository: **[Student-Performance-Prediction-AI](https://github.com/abdurrahmancce/Student-Performance-Prediction-AI)**

---

## 📚 Project Context

This project was developed as a practical machine learning application demonstrating the complete workflow from **data generation → preprocessing → model training → evaluation → model persistence → prediction → GUI deployment**.

---

## ⭐ Support

If you find this project useful:

⭐ Star the repository  
🍴 Fork the project  
🐛 Report issues  
💡 Suggest improvements  
🤝 Contribute to the project

---

<div align="center">

### 🎓 Student Performance Prediction AI

**Turning academic data into early-warning insights.**

Made with ❤️ and Python by **Abdur Rahman**

</div>
