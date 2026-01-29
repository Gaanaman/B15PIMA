import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import os
import joblib

# Set global seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Create necessary directories
os.makedirs('Results', exist_ok=True)
os.makedirs('Models', exist_ok=True)

def load_dataset(file_path='Data/PIMA_Diabetes_Source.csv'):
    """
    Loads the PIMA Indians Diabetes Dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    print("--- Loading Dataset ---")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with {df.shape[0]} samples and {df.shape[1]} columns.")
    return df

def run_eda(df):
    """
    Performs Exploratory Data Analysis and saves visualizations.
    Answers analytical questions regarding prevalence, glucose distribution,
    BMI relationships, and age impact.
    """
    print("\n--- Running Exploratory Data Analysis ---")
    
    # Prevalence of Diabetes
    outcome_counts = df['Outcome'].value_counts()
    print("Q1: What is the prevalence of diabetes in the dataset?")
    print(f"Non-diabetic (0): {outcome_counts[0]} ({outcome_counts[0]/len(df)*100:.1f}%)")
    print(f"Diabetic (1): {outcome_counts[1]} ({outcome_counts[1]/len(df)*100:.1f}%)")
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Outcome', data=df, hue='Outcome', palette='viridis', legend=False)
    plt.title('Distribution of Diabetes Outcomes')
    plt.savefig('Results/outcome_distribution.png')
    plt.close()

    # Glucose levels by Outcome
    print("\nQ2: How do glucose levels differ between diabetic and non-diabetic patients?")
    avg_glucose = df.groupby('Outcome')['Glucose'].mean()
    print(f"Average Glucose (Non-diabetic): {avg_glucose[0]:.2f}")
    print(f"Average Glucose (Diabetic): {avg_glucose[1]:.2f}")
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df[df['Outcome']==0]['Glucose'], label='Non-diabetic', fill=True)
    sns.kdeplot(df[df['Outcome']==1]['Glucose'], label='Diabetic', fill=True)
    plt.title('Glucose Level Distribution by Outcome')
    plt.legend()
    plt.savefig('Results/glucose_distribution.png')
    plt.close()

    # BMI and Glucose Relationship
    print("\nQ3: What is the relationship between BMI and Glucose across outcomes?")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='BMI', y='Glucose', hue='Outcome', data=df, alpha=0.6)
    plt.title('Glucose vs BMI by Outcome')
    plt.savefig('Results/glucose_bmi_scatter.png')
    plt.close()

    # Age impact
    print("\nHow does age relate to diabetes prevalence?")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Outcome', y='Age', data=df)
    plt.title('Age Distribution by Outcome')
    plt.savefig('Results/age_boxplot.png')
    plt.close()
    
    # Feature Correlation
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('Results/correlation_heatmap.png')
    plt.close()

def clean_data(df):
    """
    Performs data cleaning including duplicate removal and handling illogical zeros.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    print("\n--- Data Cleaning ---")
    
    # Remove Duplicate Rows
    initial_count = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} duplicate rows.")
    else:
        print("No duplicate rows found.")

    # Handle Logical Zeros (Placeholders for missing values)
    # Columns where 0 is physiologically impossible
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_fix:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            print(f"Cleaning {col}: Found {zero_count} zero values.")
            df[col] = df[col].replace(0, np.nan)
            # Impute with median (robust cleaning)
            df[col] = df[col].fillna(df[col].median())
            
    return df

def preprocess_pipeline(df):
    """
    Final preprocessing before model training.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("\n--- Data Preprocessing Pipeline ---")
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Stratified split to maintain class balance in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Fit scaler on training data ONLY to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains multiple classifiers and evaluates them using various metrics.
    
    Returns:
        tuple: (trained_models_dict, results_dataframe)
    """
    print("\n--- Model Training and Comparison ---")
    
    model_definitions = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_SEED),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED),
        "SVM": SVC(probability=True, random_state=RANDOM_SEED)
    }
    
    performance_metrics = []
    
    for name, model in model_definitions.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        results = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions),
            "Recall": recall_score(y_test, predictions),
            "F1-Score": f1_score(y_test, predictions),
            "ROC-AUC": roc_auc_score(y_test, probabilities)
        }
        performance_metrics.append(results)
        print(f"Completed evaluation for: {name}")

    results_df = pd.DataFrame(performance_metrics)
    print("\nModel Performance Comparison Table:")
    print(results_df.sort_values(by="ROC-AUC", ascending=False).to_string(index=False))
    
    return model_definitions, results_df


def plot_feature_importance(model, feature_names):
    """
    Visualizes the importance of each medical feature as determined by the model.
    """
    print("\n--- Feature Importance Analysis ---")
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importances for Diabetes Prediction')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.savefig('Results/feature_importance.png')
    plt.close()
    print("Feature importance plot saved to 'Results/feature_importance.png'.")

def plot_confusion_matrix_custom(y_true, y_pred, model_name):
    """
    Saves a heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'Results/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.close()

def save_artifacts(model, scaler):
    """
    Persists the trained model and scaler to the 'Models/' directory.
    """
    print("\n--- Saving Project Artifacts ---")
    os.makedirs('Models', exist_ok=True)
    joblib.dump(model, 'Models/best_diabetes_model.pkl')
    joblib.dump(scaler, 'Models/standard_scaler.pkl')
    print("Model saved as 'Models/best_diabetes_model.pkl'")
    print("Scaler saved as 'Models/standard_scaler.pkl'")

def predict_risk(model, scaler):
    """
    Simulates a decision-support system using a sample profile.
    """
    print("\n--- Decision-Support System Simulation ---")
    
    feature_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    # Sample data: [Pregnancies, Glucose, BP, Skin, Insulin, BMI, DPF, Age]
    sample_profile = pd.DataFrame([[2, 120, 70, 20, 80, 25.5, 0.45, 30]], columns=feature_names)
    sample_scaled = scaler.transform(sample_profile)
    
    risk_found = model.predict(sample_scaled)[0]
    risk_prob = model.predict_proba(sample_scaled)[0][1]
    
    print(f"Profile Evaluated: {sample_profile.values.tolist()[0]}")
    print(f"Assessment: {'High Concern (Diabetic Risk)' if risk_found == 1 else 'Low Concern'}")
    print(f"Probability Score: {risk_prob:.2%}")
    print("\nDISCLAIMER: This is a data-driven insight, NOT a medical diagnosis.")

def main():
    """Main execution entry point representing the complete ML lifecycle."""
    try:
        # Pipeline Start: Data Loading
        data = load_dataset()
        
        # Exploratory Phase
        run_eda(data)
        
        # Data Cleaning Phase
        data = clean_data(data)
        
        # Feature Engineering & Preprocessing
        X_tr, X_te, y_tr, y_te, trained_scaler = preprocess_pipeline(data)
        
        # Model Training & Evaluation
        trained_clfs, results_df = evaluate_models(X_tr, X_te, y_tr, y_te)
        
        # Detailed Analysis of Best Model (Random Forest)
        best_model_name = "Random Forest"
        best_model = trained_clfs[best_model_name]
        
        # Plot Confusion Matrix for best model
        y_test_pred = best_model.predict(X_te)
        plot_confusion_matrix_custom(y_te, y_test_pred, best_model_name)
        
        # Plot Feature Importance
        feature_names = [col for col in data.columns if col != 'Outcome']
        plot_feature_importance(best_model, feature_names)
        
        # Model Persistence (Deployment Readiness)
        save_artifacts(best_model, trained_scaler)
        
        # Inference / Simulation
        predict_risk(best_model, trained_scaler)
        
        print("\nEnd-to-end ML Pipeline execution completed successfully.")
        
    except Exception as e:
        print(f"\nPipeline Error occurred: {e}")

if __name__ == "__main__":
    main()
