# Presentation: Predictive Analytics for Early Diabetes Detection

## Slide 1: Project Overview
**Assignment:** DSCD 611 Final Project | **Group:** Group B15  
**Course:** DSCD 611: Programming for Data Scientists I  
**Project Title:** Machine Learning for Early-Warning Diabetes Screening  
**Institution:** University of Ghana – Legon (Comp. Sci. Dept.)  
**Instructors:** Clifford Broni-Bediako and Michael Soli  
**Group Leader:** Edward Tsatsu Akorlie | **Date:** 14th November 2025

---

## Slide 2: The Healthcare Challenge (Problem Statement)
- **Global Crisis:** Type 2 Diabetes affects 422M+ people; early detection is critical to prevent permanent organ damage and economic burden.
- **Scientific Goal:** Can we reliably predict diabetes using non-specialized health metrics?
- **Social Impact:** To provide an automated, low-cost screening tool for resource-limited community clinics in rural areas.
- **Objective:** Build a **Supervised Binary Classifier** to distinguish between 'Healthy' (0) and 'Diabetic' (1).

---

## Slide 3: Dataset Composition & Feature Rationale
- **Source:** PIMA Indians Diabetes Dataset (768 Samples | 9 Columns).
- **Predictor Variables:** 1. pregnancies, 2. Glucose, 3. Blood Pressure, 4. Skin Thickness, 5. Insulin, 6. BMI, 7. Pedigree Function, 8. Age.
- **Selection Rationale:** Targets the "Metabolic Syndrome" triad (Glucose, BMI, Hypertension).
- **Practicality:** Metrics are easy to collect without advanced hospital laboratory infrastructure.

---

## Slide 4: Key Insights from Exploratory Data Analysis (EDA)
- **Q1: Prevalence:** 34.9% of the surveyed population is diabetic, indicating a high-risk demographic needing investigation.
- **Q2: Glucose Variance:** Diabetic patients exhibit significantly higher median glucose levels (140+ mg/dL) compared to non-diabetics.
- **Q3: Correlation Proof:** Glucose and BMI show the strongest positive correlation with the outcome (Confirmed visually in our Correlation Heatmap).
- **Q4: Age Impact:** Risk significantly increases in the 30-50 age bracket within this cohort.

---

## Slide 5: Data Cleaning & Preprocessing Pipeline
- **Missing Data:** Identified "Logical Zeros"—physiologically impossible values in Glucose, BMI, and Insulin representing missing clinical data.
- **Cleaning:** Applied **Median Imputation** for robustness against outliers discovered during EDA.
- **Feature Scaling:** Used **StandardScaler** to normalize features, ensuring distance-based models (KNN, SVM) perform optimally.
- **Data Integrity:** Employed an **80/20 Stratified Split** to prevent bias and ensure results are reproducible.

---

## Slide 6: Modeling Strategy & Algorithm Comparison
We implemented and cross-evaluated five Supervised Machine Learning algorithms:
1. **Random Forest:** Top performer; uses tree ensembling for high accuracy.
2. **K-Nearest Neighbors (KNN):** Groups patients by medical similarity.
3. **SVM:** Effective in identifying complex, high-dimensional boundaries.
4. **Logistic Regression:** Provides a clinical baseline for linear risk probability.
5. **Decision Tree:** Offers clear, interpretable branch-based rules.

---

## Slide 7: Evaluated Performance & Model Selection
- **The Winner:** **Random Forest Classifier** achieved the highest overall metrics.
- **Final Metrics:** 
  - **ROC-AUC:** 0.818 (Excellent discrimination ability)
  - **Accuracy:** 77.92% | **F1-Score:** 0.66
- **Observation:** Ensemble methods significantly outperformed linear models in identifying non-linear metabolic patterns.
- **Inference:** Model correctly prioritized Glucose and BMI as the dominant signals.

---

## Slide 8: Societal Impact & Project Reflections
- **Practicality:** The pipeline provides a scalable proof-of-concept for automated triage in community health centers.
- **Lessons Learned:** "Clean Code" and preprocessing are more impactful than algorithm choice—cleaning 'Logical Zeros' was the turning point for performance.
- **Future Improvements:** Integrate genetic data for personalized medicine.
- **Conclusion:** Data Science can bridge the gap between raw medical metrics and life-saving preventative care.

---
*Group B15 | University of Ghana*
