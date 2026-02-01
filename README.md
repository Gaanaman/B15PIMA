# PIMA Indians Diabetes Prediction Project

**Course:** DSCD 611: Programming for Data Scientists | **Group:** Cohort B, Group 15

This project implements a complete Supervised Machine Learning pipeline for the early prediction of diabetes using Binary Classification. It includes an **interactive Streamlit dashboard** for data exploration and risk assessment.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Reproducibility & Installation](#reproducibility--installation)
- [Running the Analysis](#running-the-analysis)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Model Performance](#model-performance)
- [Coding Standards](#coding-standards)

---

## Overview

The goal is to predict the risk of diabetes based on medical metrics (Glucose, BMI, Age, etc.). We use the **PIMA Indians Diabetes Dataset** and compare multiple classifiers to find the most robust predictive model for community health screening.

### Key Features
- **Exploratory Data Analysis** with clinical interpretations
- **5 ML Models** compared (Random Forest, SVM, KNN, Logistic Regression, Decision Tree)
- **Interactive Streamlit Dashboard** for data exploration and risk prediction
- **Detailed Jupyter Notebook** with markdown explanations

---

## Project Structure

```
B15PIMA/
├── diabetes_analysis.py          # Main Python script (terminal executable)
├── streamlit_app.py              # Interactive Streamlit dashboard
├── exploratory_data_analysis.ipynb  # Research notebook with detailed explanations
├── requirements.txt              # Project dependencies
├── Data/
│   ├── PIMA_Diabetes_Source.csv  # Raw dataset
│   └── pima_preprocessed.csv     # Cleaned dataset (zeros imputed)
├── Models/
│   ├── best_diabetes_model.pkl   # Trained Random Forest model
│   └── standard_scaler.pkl       # Feature scaler
├── Results/                      # Visualization outputs
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── confusion_matrix_random_forest.png
│   └── ...
└── Reports/                      # Academic deliverables
```

---

## Reproducibility & Installation

### 1. Clone the repository
```bash
git clone https://github.com/gyau-k/B15PIMA.git
cd B15PIMA
```

### 2. Setup virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Analysis

### Python Script (Terminal)
Execute the full pipeline (EDA, Preprocessing, Training, and Simulation):
```bash
python3 diabetes_analysis.py
```

This will:
- Load and clean the dataset
- Generate visualizations in `Results/`
- Train and compare 5 ML models
- Save the best model to `Models/`
- Run a sample prediction

---

## Streamlit Dashboard

**🌐 Live Demo:** [https://b15pima.streamlit.app/](https://b15pima.streamlit.app/)

Launch the interactive web dashboard locally:
```bash
streamlit run streamlit_app.py
```

### Dashboard Features
| Feature | Description |
|---------|-------------|
| **Key Metrics** | Diabetes prevalence, glucose, BMI, age statistics |
| **Research Questions** | Interactive visualizations for 4 clinical questions |
| **Correlation Heatmap** | Feature relationships at a glance |
| **Risk Prediction** | Input patient data and get diabetes risk assessment |
| **Data Explorer** | Browse the raw dataset |

---

## Model Performance

Our analysis compared 5 algorithms. The **Random Forest** model was selected as the best performer:

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **Random Forest** | **77.9%** | **0.818** |
| K-Nearest Neighbors | 75.3% | 0.789 |
| Support Vector Machine | 74.0% | 0.796 |
| Logistic Regression | 70.8% | 0.813 |
| Decision Tree | 68.2% | 0.636 |

### Clinical Interpretation
- **ROC-AUC of 0.82**: Good discriminative ability for screening purposes
- **Top predictors**: Glucose, BMI, Age, Diabetes Pedigree Function
- **Use case**: Screening tool to flag high-risk patients for clinical evaluation

---

## Coding Standards

- **Modularity**: Code is split into logical units (loading, EDA, preprocessing, evaluation)
- **Docstrings**: All functions include Google-style docstrings
- **Random States**: A global `RANDOM_SEED = 42` ensures identical results on every run
- **Robustness**: Includes error handling and median imputation for missing medical data
- **Clean UI**: Streamlit dashboard follows Seattle Weather demo design principles

---

## Team

**Cohort B · Group 15** — University of Ghana, Legon

| Role | Name |
|------|------|
| Group Leader | Edward Tsatsu Akorlie |
| UI/UX | Daniel K. Adotey |
| Member | Kwame Ofori-Gyau |
| Member | Francis A. Sarbeng |
| Member | Caleb A. Mensah |

---

*Developed for DSCD 611: Programming for Data Scientists | February 2026*
