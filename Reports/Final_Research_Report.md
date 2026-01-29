# FINAL PROJECT REPORT: Early Prediction of Diabetes

**Assignment:** DSCD 611 Final Project Report  
**Course:** DSCD 611: Programming for Data Scientists I  
**Project Title:** Predictive Modeling for Type 2 Diabetes Melitus in Pima Indian Populations  
**Group:** Group B15  
**Institution:** Department of Computer Science, University of Ghana – Legon  
**Instructors:** Clifford Broni-Bediako and Michael Soli  
**Date:** 14th November 2025

**Group Leader:** Edward Tsatsu Akorlie (ID: 22424530)  
**Members:**  
- Daniel Kpakpo Adotey (ID: 22424924)  
- Kwame Ofori-Gyau (ID: 22424324)  
- Francis Aboraa Sarbeng (ID: 22424635)  
- Caleb Abakah Mensah (ID: 22424188)  

---

## 1. Topic Studied
The project focuses on the development of a **Supervised Machine Learning** pipeline to address a **Binary Classification** problem: identifying signs of Type 2 Diabetes Mellitus (T2DM). Using a clinical dataset of female patients of Pima Indian heritage, the study investigates which metabolic markers—such as glucose concentration and body mass index (BMI)—serve as the most reliable early-warning indicators for disease onset.

## 2. What is Known of the Topic
Type 2 Diabetes is a chronic condition characterized by insulin resistance. Medical literature establishes that genetics, lifestyle, and obesity are the primary drivers. Studies from health organizations like the National Institute of Diabetes and Digestive and Kidney Diseases have highlighted that Pima Indians have one of the highest prevalences of diabetes globally. Conventional screening relies on glucose tolerance tests, which can be resource-intensive in rural areas. Machine learning offers a way to augment these clinical tests by identifying patterns in non-specialized health metrics.

## 3. Why the Topic is Interesting, Relevant, or Important
Diabetes is a global health crisis, affecting over 422 million people. It is a leading cause of blindness, kidney failure, heart attacks, and stroke. This project is particularly relevant because:
- **Societal Impact:** Early detection can prevent permanent organ damage and reduce the economic burden on healthcare systems.
- **Resource Management:** In low-resource settings, an automated tool can help triage high-risk patients for further clinical testing.
- **Scientific Value:** It demonstrates how predictive modeling can bridge the gap between raw data and actionable health insights.

## 4. Description of Data Used
The **PIMA Indians Diabetes Dataset** consists of 768 samples with 9 medical features:
- **Pregnancies:** Number of times pregnant.
- **Glucose:** Plasma glucose concentration (2 hours in an oral glucose tolerance test).
- **Blood Pressure:** Diastolic blood pressure (mm Hg).
- **Skin Thickness:** Triceps skin fold thickness (mm).
- **Insulin:** 2-Hour serum insulin (mu U/ml).
- **BMI:** Body mass index (weight in kg/(height in m)^2).
- **Diabetes Pedigree Function:** A function which scores likelihood of diabetes based on family history.
- **Age:** Age in years.
- **Outcome:** Binary target (0: No Diabetes, 1: Diabetes).

### 4.1 Rationale for Feature Selection
The selection of these 8 features was driven by a combination of clinical theory and statistical evidence:
1.  **Clinical Relevance:** Features like Glucose, BMI, and Insulin are part of the 'Metabolic Syndrome' markers used globally by clinicians to screen for insulin resistance.
2.  **Demographic Impact:** Pregnancy history and Age are statistically significant risk factors for diabetes onset in adult women.
3.  **Statistical Evidence:** Our EDA (Section 5) confirmed that Glucose and BMI exhibit high correlation with the target variable, making them computationally essential for high-accuracy modeling.
4.  **Practicality:** All selected features are non-specialized metrics that can be collected in community health centers without advanced hospital equipment.

## 5. How the Project was Done (Tools and Methods)

### 5.1 Analytical Research Questions
To guide our exploration, we formulated and addressed the following four questions:
1.  **Prevalence:** What is the distribution of diabetes outcomes within the Pima Indian demographic?
2.  **Metabolic Correlation:** How significantly do plasma glucose levels vary between diabetic and non-diabetic cohorts?
3.  **Risk Interactions:** What is the visual relationship between BMI and Glucose, and how does it shift across outcomes?
4.  **Demographic Impact:** Does age show a statistically observable correlation with diabetes risk in this patient group?

The project utilized an end-to-end Python pipeline centered on the following ecosystem:
- **Programming Environment:** Python 3.12, Jupyter Notebooks.
- **Libraries:** `Pandas` (data manipulation), `NumPy` (numerical arrays), `Matplotlib/Seaborn` (visualizations), and `Scikit-Learn` (modeling).
- **Pipeline Stages:**
  1. **Data Cleaning:** Removal of duplicates and identifying "Logical Zeros".
  2. **EDA:** Statistical analysis and visualization of feature distributions.
  3. **Preprocessing:** Median imputation and `StandardScaler` normalization.
  4. **Modeling:** Comparative analysis of 5 classifiers.
  5. **Interpretability:** Feature importance mapping and confusion matrix generation.

## 6. Results and Societal Impact
The models were evaluated based on their ability to correctly classify patients.

### 6.1 Performance Metrics
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 0.7792 | 0.7174 | 0.6111 | 0.6600 | 0.8179 |
| KNN | 0.7532 | 0.6600 | 0.6111 | 0.6346 | 0.7886 |
| SVM | 0.7403 | 0.6522 | 0.5556 | 0.6000 | 0.7964 |
| Logistic Reg. | 0.7078 | 0.6000 | 0.5000 | 0.5455 | 0.8130 |
| Decision Tree | 0.6818 | 0.5532 | 0.4815 | 0.5149 | 0.6357 |

- **Key Findings:** Glucose and BMI emerged as the statistically dominant predictors.
- **Societal Impact:** Early-detection can potentially save thousands of lives through preventative interventions.

## 7. Team Member Contributions
- **Edward Tsatsu Akorlie (Leader):** Overall project architect, pipeline development (`diabetes_analysis.py`), and Random Forest hyperparameter tuning.
- **Daniel Kpakpo Adotey:** Lead on data cleaning, outlier detection, and missing value imputation logic.
- **Kwame Ofori-Gyau:** Conducted comprehensive dataset exploration in Jupyter and led the statistical analysis phase.
- **Francis Aboraa Sarbeng:** Lead on visualization design—created the correlation heatmaps and final results graphics.
- **Caleb Abakah Mensah:** Technical lead for the Reports, Project Proposal, and PPT presentation slides.

## 8. Reflections on the Project
Based on the analysis of the **PIMA Indians Diabetes Dataset**, we conclude that metabolic features like **Glucose** and **high BMI** are the most significant clinical predictors of diabetes. The team learned that data cleaning is critical; the "Logical Zeros" would have skewed results if left unhandled. We also reflected on the ethical limitations regarding generalizability. Overall, the project successfully demonstrated that machine learning can reliably identify risk patterns using non-invasive medical metrics, providing a robust baseline for early screening in high-risk populations.

## 9. References
1. WHO. (2024). *Global Diabetes Compact*. World Health Organization.
2. National Institute of Diabetes and Digestive and Kidney Diseases. (1990). *Pima Indians Diabetes Dataset*. UCI Machine Learning Repository.
3. Broni-Bediako, C. & Soli, M. (2025). *Course Materials: DSCD 611 - Programming for Data Scientists I*. University of Ghana.

---
*Submitted as the final project requirement for DSCD 611.*
