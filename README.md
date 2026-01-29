# PIMA Indians Diabetes Prediction Project
**Course:** DSCD 611: Programming for Data Scientists I | **Group:** Group B15

This project implements a complete **Supervised Machine Learning** pipeline for the early prediction of diabetes using **Binary Classification**. It is designed to be **clean**, **modular**, and **completely reproducible**.

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Reproducibility & Installation](#reproducibility--installation)
4. [Running the Analysis](#running-the-analysis)
5. [Model Performance](#model-performance)

## Overview
The goal is to predict the risk of diabetes based on medical metrics (Glucose, BMI, Age, etc.). We use the PIMA Indians Diabetes Dataset and compare multiple classifiers to find the most robust predictive model for community health screening.

## Project Structure
- `diabetes_analysis.py`: Main Python script implementing the end-to-end ML workflow.
- `Exploratory_Analysis.ipynb`: Research notebook with detailed discovery and iterative modeling.
- `requirements.txt`: Project dependencies for environment reproducibility.
- `Data/`: Contains the `PIMA_Diabetes_Source.csv` raw dataset.
- `Models/`: Persisted model artifacts (serialized `.pkl` files).
- `Results/`: Visual outputs (Correlation heatmaps, Feature importance, Confusion matrices).
- `Reports/`: Final academic deliverables (Research Report, Proposal, Slides).

## Reproducibility & Installation

1. **Clone/Download** the repository.
2. **Setup virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Data Acquisition**: The script expects the source dataset at `Data/PIMA_Diabetes_Source.csv`.

## Running the Analysis
To execute the full pipeline (EDA, Preprocessing, Training, and Simulation):
```bash
python3 diabetes_analysis.py
```

## Model Performance
Our analysis compared 5 algorithms. The **Random Forest** model was selected as the best performer:

| Model | Accuracy | ROC-AUC |
| :--- | :--- | :--- |
| **Random Forest** | **77.9%** | **0.818** |
| K-Nearest Neighbors | 75.3% | 0.789 |
| Support Vector Machine | 74.0% | 0.796 |
| Logistic Regression | 70.8% | 0.813 |
| Decision Tree | 68.2% | 0.636 |

## Coding Standards
- **Modularity**: Code is split into logical units (loading, EDA, preprocessing, evaluation).
- **Docstrings**: All functions include Google-style docstrings.
- **Random States**: A global `RANDOM_SEED` (42) ensures identical results on every run.
- **Robustness**: Includes error handling and median imputation for missing medical data.

---
*Developed by Group B15 for DSCD 611 | University of Ghana – Legon*
