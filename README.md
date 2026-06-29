# 🎓 Student Performance Prediction — End-to-End ML Project

A production-style machine learning project that predicts student exam performance based on demographic and academic inputs. Built with a modular pipeline architecture, Flask web app, and full ML lifecycle coverage — from EDA to deployment-ready inference.

---

## 📁 Project Structure

```
DataScience_end_to_end_project/
├── artifact/                    # Auto-generated outputs from pipeline runs
│   ├── data.csv                 # Raw ingested data
│   ├── train.csv                # Train split
│   ├── test.csv                 # Test split
│   ├── preprocessor.pkl         # Saved ColumnTransformer
│   └── model.pkl                # Best trained model
│
├── notebook/                    # Exploratory work
│   ├── data/study.csv           # Raw dataset
│   ├── 1_EDA.ipynb              # Exploratory Data Analysis
│   └── model-training.ipynb     # Baseline model experiments
│
├── src/                         # Core source package
│   ├── components/
│   │   ├── data_ingestion.py    # Reads raw data, creates train/test splits
│   │   ├── data_transformation.py  # Preprocessing pipelines (scaling, encoding)
│   │   └── model_trainer.py     # Model selection, training, evaluation
│   ├── pipelines/
│   │   ├── train.py             # Orchestrates full training pipeline
│   │   └── predict.py           # Inference pipeline for single predictions
│   ├── exception.py             # Custom exception handling
│   ├── logger.py                # Centralized logging setup
│   └── utils.py                 # Shared utilities (save/load pkl, evaluate models)
│
├── templates/                   # Flask HTML templates
│   ├── index.html               # Landing page
│   └── home.html                # Prediction form
│
├── logs/                        # Runtime logs (auto-generated)
├── catboost_info/               # CatBoost training metadata
├── app.py                       # Flask web application
├── setup.py                     # Package setup
└── requirements.txt             # Dependencies
```

---

## 🧠 Problem Statement

Predict a student's **math score** based on features like gender, race/ethnicity, parental education, lunch type, test preparation course, reading score, and writing score.

**Type:** Regression  
**Target:** `math_score` (continuous)

---

## ⚙️ ML Pipeline

```
Raw CSV
  └─► Data Ingestion         → train.csv / test.csv
        └─► Data Transformation  → preprocessor.pkl (StandardScaler + OneHotEncoder)
              └─► Model Trainer      → model.pkl (best model via R² comparison)
```

### Models Evaluated

- Random Forest Regressor
- Decision Tree Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- CatBoost Regressor
- AdaBoost Regressor
- Linear Regression
<!-- - Ridge / Lasso -->

Best model is selected based on **R² score** on test set and saved to `artifact/model.pkl`.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/DataScience_end_to_end_project.git
cd DataScience_end_to_end_project
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Training Pipeline

```bash
python src/pipelines/train.py
```

Artifacts (`model.pkl`, `preprocessor.pkl`, `train.csv`, `test.csv`) will be saved to `artifact/`.

### 5. Launch Web App

```bash
python app.py
```

Open `http://localhost:5000` in your browser. Fill in student details and get a predicted math score.

---

## 🌐 Web Application

Built with **Flask**. The prediction form (`home.html`) takes:

| Feature                     | Type        |
| --------------------------- | ----------- |
| Gender                      | Categorical |
| Race/Ethnicity              | Categorical |
| Parental Level of Education | Categorical |
| Lunch Type                  | Categorical |
| Test Preparation Course     | Categorical |
| Reading Score               | Numeric     |
| Writing Score               | Numeric     |

The `/predict` route calls `PredictPipeline` → loads `preprocessor.pkl` + `model.pkl` → returns predicted math score.

---

## 📊 EDA Highlights

Covered in `notebook/1_EDA.ipynb`:

- Target distribution and outlier analysis
- Feature correlations (reading/writing scores highly correlated with math score)
- Group-wise performance breakdown by gender, parental education, and test prep

---

## 🛠 Tech Stack

| Layer         | Tools                           |
| ------------- | ------------------------------- |
| Language      | Python 3.x                      |
| ML Libraries  | scikit-learn, XGBoost, CatBoost |
| Web Framework | Flask                           |
| Data          | pandas, numpy                   |
| Logging       | Python `logging` module         |
| Serialization | `dill` / `pickle`               |
| Notebook      | Jupyter                         |

---

## 📝 Logging & Exception Handling

- All pipeline stages log to `logs/` with timestamps via `src/logger.py`
- Custom `CustomException` in `src/exception.py` captures file name and line number for clean tracebacks

---

## 📦 Package Setup

The `src/` directory is installable as a local package via `setup.py`:

```bash
pip install -e .
```

This enables clean imports like `from src.components.model_trainer import ModelTrainer` across the project.

---

## 🙋 Author

**Gautam Mishra**  
Data Science & AI/ML | Fresher  
[LinkedIn](https://linkedin.com/in/) • [GitHub](https://github.com/)

---

## 📄 License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
