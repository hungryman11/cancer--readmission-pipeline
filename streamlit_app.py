# streamlit_app.py - FINAL VERSION with GridSearchCV Hyperparameter Tuning + Ensemble Model

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --------------------------------------------------------------
st.set_page_config(page_title="Cancer Readmission Predictor", layout="wide")
st.title("🏥 Cancer Patient 30-Day Readmission Prediction")
st.markdown("Advanced version with **hyperparameter tuning** and **ensemble modeling**.")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

st.sidebar.markdown("""
**Upload Tips:**
- Save as **CSV UTF-8**
- Mixed data is cleaned automatically
""")

# --------------------------------------------------------------
# 1. Synthetic Demo Data
# --------------------------------------------------------------
if uploaded_file is None:
    st.info("No file uploaded — generate demo data below.")
    if st.button("Generate & Download Synthetic Demo Data"):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=10000, n_features=20, weights=[0.85, 0.15], random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['readmitted_30days'] = y

        np.random.seed(42)
        df['patient_id'] = range(1, len(df) + 1)
        df['age'] = np.random.normal(65, 12, len(df)).clip(18, 90).astype(int)
        df['cancer_stage'] = np.random.choice(['Stage I', 'Stage II', 'Stage III', 'Stage IV'], size=len(df), p=[0.25, 0.35, 0.25, 0.15])
        df['chemo_cycles'] = np.random.poisson(4, len(df))
        df['length_of_stay'] = np.random.lognormal(1.8, 0.7, len(df)).astype(int) + 1
        df['num_medications'] = np.random.poisson(8, len(df)) + 1
        df['previous_admissions'] = np.random.poisson(1.5, len(df))
        df['icu_stay'] = np.random.binomial(1, 0.2, len(df))
        df['emergency_admission'] = np.random.binomial(1, 0.4, len(df))
        df['hemoglobin'] = np.random.normal(12, 2, len(df)).clip(6, 18)
        df['albumin'] = np.random.normal(4.0, 0.6, len(df)).clip(2, 5.5)

        high_risk = df['readmitted_30days'] == 1
        df.loc[high_risk, 'age'] = (df.loc[high_risk, 'age'] + np.random.normal(8, 5, high_risk.sum())).clip(18, 95).astype(int)
        df.loc[high_risk, 'length_of_stay'] += np.random.poisson(5, high_risk.sum())

        csv = df.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="synthetic_cancer_data.csv">Download Synthetic Data</a>'
        st.markdown(href, unsafe_allow_html=True)
    st.stop()

# --------------------------------------------------------------
# 2. Load & Clean Data (same robust version as before)
# --------------------------------------------------------------
@st.cache_data
def load_data(file):
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'iso-8859-1', 'latin1']
    for enc in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc, low_memory=False)
        except Exception:
            continue
    file.seek(0)
    st.warning("Using fallback 'latin1'")
    return pd.read_csv(file, encoding='latin1', low_memory=False, errors='replace')

df_raw = load_data(uploaded_file)
st.success(f"Loaded {df_raw.shape[0]} rows")

# Clean mixed data
cancer_markers = ['ID', 'Radius_mean', 'Diagnosis']
cancer_start_idx = None
for i, col in enumerate(df_raw.columns):
    if col in cancer_markers:
        cancer_start_idx = i
        break

if cancer_start_idx is not None:
    st.info(f"Removing junk columns after index {cancer_start_idx}")
    df = df_raw.iloc[:, :cancer_start_idx].copy()
else:
    df = df_raw.copy()

if 'Row_ID' in df.columns:
    df = df.dropna(subset=['Row_ID']).copy()

# Patient ID detection
possible_id_cols = ['Patient_ID', 'Patientt_ID', 'SUBJECT_ID', 'subject_id', 'PATIENT_ID', 'Hadm_ID', 'HADM_ID']
patient_id_col = next((col for col in possible_id_cols if col in df.columns), None)
if patient_id_col is None:
    st.error("No patient ID column found.")
    st.stop()

df = df.rename(columns={patient_id_col: 'Patient_ID'})

# Parse dates
def parse_admission_datetime(row, date_col, year_col):
    if pd.isna(row.get(date_col)) or pd.isna(row.get(year_col)):
        return pd.NaT
    try:
        parts = str(row[date_col]).split(' ')
        date_part = parts[0]
        time_part = parts[1] if len(parts) > 1 else '00:00'
        mm, dd, yy = date_part.split('/')
        full_year = int(row[year_col])
        return pd.to_datetime(f"{mm}/{dd}/{full_year} {time_part}", format='%m/%d/%Y %H:%M', errors='coerce')
    except:
        return pd.NaT

df['Admit_time'] = df.apply(lambda row: parse_admission_datetime(row, 'Admitted_date', 'Admitted_year'), axis=1)
df['Disch_time'] = df.apply(lambda row: parse_admission_datetime(row, 'Disch_date', 'Disch_year'), axis=1)

df = df.dropna(subset=['Admit_time', 'Disch_time', 'Patient_ID']).copy()
df = df.sort_values(['Patient_ID', 'Admit_time']).reset_index(drop=True)

# Create target
if 'readmitted_30days' not in df.columns:
    st.info("Generating target...")
    df['readmitted_30days'] = 0
    for pid in df['Patient_ID'].unique():
        patient_data = df[df['Patient_ID'] == pid]
        if len(patient_data) > 1:
            for i in range(len(patient_data) - 1):
                if pd.notna(patient_data.iloc[i]['Disch_time']) and pd.notna(patient_data.iloc[i+1]['Admit_time']):
                    days = (patient_data.iloc[i+1]['Admit_time'] - patient_data.iloc[i]['Disch_time']).days
                    if 0 < days <= 30:
                        df.loc[patient_data.index[i], 'readmitted_30days'] = 1
    st.success(f"Target created: {df['readmitted_30days'].sum()} readmissions")

# Feature engineering (same as before)
def engineer_features(data):
    df = data.copy()
    if 'age' in df.columns:
        df['is_elderly'] = (df['age'] >= 65).astype(int)
    if 'cancer_stage' in df.columns:
        df['is_advanced_stage'] = df['cancer_stage'].astype(str).str.contains('III|IV|3|4', case=False, na=False).astype(int)
    if 'chemo_cycles' in df.columns:
        df['has_chemotherapy'] = (df['chemo_cycles'] > 0).astype(int)
    if 'length_of_stay' in df.columns:
        df['prolonged_stay'] = (df['length_of_stay'] > 7).astype(int)
    if 'num_medications' in df.columns:
        df['polypharmacy'] = (df['num_medications'] >= 10).astype(int)
    if 'hemoglobin' in df.columns:
        df['anemia'] = (df['hemoglobin'] < 10).astype(int)
    if 'albumin' in df.columns:
        df['hypoalbuminemia'] = (df['albumin'] < 3.5).astype(int)

    risk_cols = ['is_elderly', 'is_advanced_stage', 'has_chemotherapy', 'prolonged_stay', 'polypharmacy', 'anemia', 'hypoalbuminemia']
    risk_cols = [c for c in risk_cols if c in df.columns]
    if risk_cols:
        df['composite_risk_score'] = df[risk_cols].sum(axis=1)
    return df

df_final = engineer_features(df)

# --------------------------------------------------------------
# 6. Modeling with GridSearchCV + Ensemble
# --------------------------------------------------------------
target = 'readmitted_30days'
X = df_final.select_dtypes(include=[np.number]).drop(columns=[target, 'Patient_ID', 'Row_ID', 'Hadm_ID'], errors='ignore')
y = df_final[target]

if X.isna().any().any():
    st.warning("Imputing missing values with median.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Base estimators with GridSearchCV
st.header("Hyperparameter Tuning with GridSearchCV")

# Logistic Regression Grid
lr_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])
lr_grid = {'model__C': [0.1, 1, 10], 'model__penalty': ['l2']}
lr_gs = GridSearchCV(lr_pipe, lr_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Random Forest Grid
rf_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestClassifier(random_state=42))
])
rf_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5]
}
rf_gs = GridSearchCV(rf_pipe, rf_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Gradient Boosting Grid
gb_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model', GradientBoostingClassifier(random_state=42))
])
gb_grid = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.01, 0.1],
    'model__max_depth': [3, 5]
}
gb_gs = GridSearchCV(gb_pipe, gb_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Train tuned models
with st.spinner("Running GridSearchCV (this may take a few minutes)..."):
    lr_gs.fit(X_train, y_train)
    rf_gs.fit(X_train, y_train)
    gb_gs.fit(X_train, y_train)

tuned_models = {
    "Tuned Logistic Regression": lr_gs.best_estimator_,
    "Tuned Random Forest": rf_gs.best_estimator_,
    "Tuned Gradient Boosting": gb_gs.best_estimator_
}

# Ensemble (Voting Classifier on tuned models)
ensemble = VotingClassifier(
    estimators=[
        ('lr', lr_gs.best_estimator_),
        ('rf', rf_gs.best_estimator_),
        ('gb', gb_gs.best_estimator_)
    ],
    voting='soft'  # Uses predicted probabilities
)
ensemble.fit(X_train, y_train)
tuned_models["Ensemble (Voting)"] = ensemble

# Evaluate all
results = {}
predictions = {}
for name, model in tuned_models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    results[name] = {'AUC': auc(fpr, tpr)}
    predictions[name] = {'prob': y_prob, 'pred': y_pred, 'fpr': fpr, 'tpr': tpr, 'precision': precision, 'recall': recall}

st.success("Hyperparameter tuning and ensemble training complete!")

# --------------------------------------------------------------
# 7. Model Comparison Table
# --------------------------------------------------------------
st.header("Model Comparison (After Tuning)")
comparison_df = pd.DataFrame({
    name: {
        'AUC-ROC': results[name]['AUC'],
        'Best Params': str(model.named_steps['model'].get_params() if 'named_steps' in dir(model) else "Ensemble")
    } for name, model in tuned_models.items()
}).T.round(4)

st.dataframe(comparison_df.style.highlight_max(subset=['AUC-ROC'], axis=0))

best_name = comparison_df['AUC-ROC'].idxmax()
st.success(f"Best Model: **{best_name}** (AUC-ROC = {comparison_df.loc[best_name, 'AUC-ROC']:.4f})")

# Visualizations (same layout)
col1, col2 = st.columns(2)
with col1:
    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for name, pred in predictions.items():
        ax.plot(pred['fpr'], pred['tpr'], label=f"{name} (AUC={results[name]['AUC']:.3f})")
    ax.plot([0,1],[0,1],'k--')
    ax.legend()
    st.pyplot(fig)

    st.subheader("Precision-Recall Curves")
    fig, ax = plt.subplots()
    for name, pred in predictions.items():
        ap = average_precision_score(y_test, pred['prob'])
        ax.plot(pred['recall'], pred['precision'], label=f"{name} (AP={ap:.3f})")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader(f"Confusion Matrix — {best_name}")
    cm = confusion_matrix(y_test, predictions[best_name]['pred'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.subheader("Feature Importance (Best Tree Model)")
    if "Forest" in best_name or "Boosting" in best_name or "Ensemble" in best_name:
        # Use Gradient Boosting for importance (usually strong)
        model_for_imp = tuned_models.get("Tuned Gradient Boosting", tuned_models["Tuned Random Forest"])
        importances = model_for_imp.named_steps['model'].feature_importances_
        imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False).head(10)
        fig, ax = plt.subplots()
        sns.barplot(data=imp_df, x='Importance', y='Feature', ax=ax)
        st.pyplot(fig)

st.download_button("Download Comparison", comparison_df.to_csv(), "tuned_model_comparison.csv")
st.balloons()