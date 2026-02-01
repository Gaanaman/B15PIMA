"""
DSCD 611: Predictive Analytics for Early Diabetes Detection
Interactive Streamlit Dashboard — Group B15
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="PIMA Diabetes Analytics",
    page_icon="🩺",
    layout="wide",
)

# ============================================
# DATA LOADING
# ============================================
@st.cache_data
def load_data():
    """Load and cache the preprocessed diabetes dataset."""
    return pd.read_csv('Data/pima_preprocessed.csv')

@st.cache_resource
def load_model():
    """Load the trained model and scaler."""
    model_path = 'Models/best_diabetes_model.pkl'
    scaler_path = 'Models/standard_scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

df = load_data()
model, scaler = load_model()

# ============================================
# HEADER
# ============================================
"""
# 🩺 PIMA Diabetes Analytics

Explore the PIMA Indians Diabetes Dataset and assess diabetes risk using Machine Learning.
"""

""  # vertical space

# ============================================
# KEY METRICS
# ============================================
diabetic_count = df[df['Class'] == 1].shape[0]
non_diabetic_count = df[df['Class'] == 0].shape[0]
diabetic_pct = (diabetic_count / df.shape[0]) * 100

avg_glucose_diabetic = df[df['Class'] == 1]['Glucose'].mean()
avg_glucose_non_diabetic = df[df['Class'] == 0]['Glucose'].mean()

avg_bmi_diabetic = df[df['Class'] == 1]['BMI'].mean()
avg_bmi_non_diabetic = df[df['Class'] == 0]['BMI'].mean()

avg_age_diabetic = df[df['Class'] == 1]['Age'].mean()
avg_age_non_diabetic = df[df['Class'] == 0]['Age'].mean()

## Summary Metrics
with st.container():
    cols = st.columns(4)
    
    with cols[0]:
        st.metric(
            "Diabetes Prevalence",
            f"{diabetic_pct:.1f}%",
            f"{diabetic_count} of {df.shape[0]} cases"
        )
    
    with cols[1]:
        # Glucose: Normal <100, Prediabetes 100-125, Diabetes ≥126
        glucose_status = "Diabetic range (≥126)" if avg_glucose_diabetic >= 126 else "Elevated"
        st.metric(
            "Avg Glucose (Diabetic)",
            f"{avg_glucose_diabetic:.0f} mg/dL",
            glucose_status,
            delta_color="inverse"  # Red since this is elevated
        )
    
    with cols[2]:
        # BMI: Normal 18.5-24.9, Overweight 25-29.9, Obese ≥30
        if avg_bmi_diabetic >= 30:
            bmi_status = "Obese (≥30)"
        elif avg_bmi_diabetic >= 25:
            bmi_status = "Overweight (25-30)"
        else:
            bmi_status = "Normal"
        st.metric(
            "Avg BMI (Diabetic)",
            f"{avg_bmi_diabetic:.1f}",
            bmi_status,
            delta_color="inverse" if avg_bmi_diabetic >= 25 else "normal"
        )
    
    with cols[3]:
        st.metric(
            "Avg Age (Diabetic)",
            f"{avg_age_diabetic:.0f} yrs",
            f"vs {avg_age_non_diabetic:.0f} (no age threshold)",
            delta_color="off"
        )

""
""

# ============================================
# RESEARCH QUESTIONS
# ============================================
"""
## Research Questions
"""

OUTCOMES = df['Class'].unique().tolist()
selected_outcomes = st.pills(
    "Filter by outcome",
    options=["All", "Non-Diabetic (0)", "Diabetic (1)"],
    default="All"
)

if selected_outcomes == "Non-Diabetic (0)":
    plot_df = df[df['Class'] == 0]
elif selected_outcomes == "Diabetic (1)":
    plot_df = df[df['Class'] == 1]
else:
    plot_df = df

# Charts Row 1
cols = st.columns([1, 2])

with cols[0].container(border=True):
    "### Diabetes Prevalence"
    
    prevalence_chart = alt.Chart(df).mark_arc(innerRadius=40).encode(
        theta=alt.Theta("count():Q"),
        color=alt.Color(
            "Class:N",
            scale=alt.Scale(domain=[0, 1], range=["#4CAF50", "#f44336"]),
            legend=alt.Legend(title="Outcome", orient="bottom")
        ),
        tooltip=["Class:N", "count():Q"]
    ).properties(height=250)
    
    st.altair_chart(prevalence_chart, use_container_width=True)

with cols[1].container(border=True):
    "### Glucose Distribution by Outcome"
    
    glucose_chart = alt.Chart(plot_df).transform_density(
        'Glucose',
        as_=['Glucose', 'density'],
        groupby=['Class']
    ).mark_area(opacity=0.5).encode(
        x=alt.X('Glucose:Q', title='Glucose Level (mg/dL)'),
        y=alt.Y('density:Q', title='Density'),
        color=alt.Color(
            'Class:N',
            scale=alt.Scale(domain=[0, 1], range=["#4CAF50", "#f44336"]),
            legend=alt.Legend(orient="bottom")
        )
    ).properties(height=250)
    
    st.altair_chart(glucose_chart, use_container_width=True)

# Charts Row 2
cols = st.columns(2)

with cols[0].container(border=True):
    "### BMI vs Glucose"
    
    scatter_chart = alt.Chart(plot_df).mark_circle(size=40, opacity=0.6).encode(
        x=alt.X('BMI:Q', scale=alt.Scale(zero=False)),
        y=alt.Y('Glucose:Q', scale=alt.Scale(zero=False)),
        color=alt.Color(
            'Class:N',
            scale=alt.Scale(domain=[0, 1], range=["#4CAF50", "#f44336"]),
            legend=alt.Legend(orient="bottom")
        ),
        tooltip=['BMI:Q', 'Glucose:Q', 'Age:Q', 'Class:N']
    ).properties(height=300).interactive()
    
    st.altair_chart(scatter_chart, use_container_width=True)

with cols[1].container(border=True):
    "### Age Distribution"
    
    age_chart = alt.Chart(plot_df).mark_boxplot(extent='min-max').encode(
        x=alt.X('Class:N', title='Outcome', axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Age:Q', title='Age (years)'),
        color=alt.Color(
            'Class:N',
            scale=alt.Scale(domain=[0, 1], range=["#4CAF50", "#f44336"]),
            legend=None
        )
    ).properties(height=300)
    
    st.altair_chart(age_chart, use_container_width=True)

""

# ============================================
# CORRELATION HEATMAP
# ============================================
cols = st.columns([2, 1])

with cols[0].container(border=True):
    "### Feature Correlations"
    
    corr_df = df.corr().reset_index().melt('index')
    corr_df.columns = ['Feature1', 'Feature2', 'Correlation']
    
    heatmap = alt.Chart(corr_df).mark_rect().encode(
        x=alt.X('Feature1:N', title=None),
        y=alt.Y('Feature2:N', title=None),
        color=alt.Color(
            'Correlation:Q',
            scale=alt.Scale(scheme='redblue', domain=[-1, 1])
        ),
        tooltip=['Feature1', 'Feature2', alt.Tooltip('Correlation:Q', format='.2f')]
    ).properties(height=350)
    
    text = alt.Chart(corr_df).mark_text(fontSize=9).encode(
        x='Feature1:N',
        y='Feature2:N',
        text=alt.Text('Correlation:Q', format='.2f'),
        color=alt.condition(
            abs(alt.datum.Correlation) > 0.4,
            alt.value('white'),
            alt.value('black')
        )
    )
    
    st.altair_chart(heatmap + text, use_container_width=True)

with cols[1].container(border=True):
    "### Key Insights"
    
    st.markdown("""
    **Top correlations with diabetes (Class):**
    - Glucose: **0.49** (strongest)
    - BMI: **0.31**
    - Age: **0.24**
    - Pregnancies: **0.22**
    
    **Observations:**
    - Glucose is the primary predictor
    - Data is preprocessed (zeros imputed)
    - Age and Pregnancies show moderate correlation
    """)

""

# ============================================
# PREDICTION TOOL
# ============================================
"""
## Diabetes Risk Prediction

Enter patient data to predict diabetes risk using our Random Forest model.
"""

if model is not None and scaler is not None:
    with st.form("prediction_form"):
        cols = st.columns(4)
        
        with cols[0]:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
            skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
        
        with cols[1]:
            glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=250.0, value=120.0)
            insulin = st.number_input("Insulin (μU/mL)", min_value=0.0, max_value=900.0, value=80.0)
        
        with cols[2]:
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=130.0, value=72.0)
            bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        
        with cols[3]:
            dpf = st.number_input("Diabetes Pedigree", min_value=0.0, max_value=2.5, value=0.5)
            age = st.number_input("Age", min_value=21, max_value=100, value=30)
        
        submitted = st.form_submit_button("Predict Risk", use_container_width=True)
    
    if submitted:
        input_features = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age
        ]])
        
        input_scaled = scaler.transform(input_features)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        if prediction == 1:
            st.error(f"**High Risk** — Probability: {probability:.1%}")
        else:
            st.success(f"**Low Risk** — Probability: {probability:.1%}")
        
        st.caption("⚕️ *This is a data-driven assessment, not a medical diagnosis.*")
else:
    st.warning("Model not found. Run `python diabetes_analysis.py` first.")

""

# ============================================
# RAW DATA
# ============================================
with st.expander("View raw data"):
    st.dataframe(df, use_container_width=True)

""

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("**Cohort B · Group 15** — PIMA Diabetes Analytics | DSCD 611: Programming for Data Scientists")
