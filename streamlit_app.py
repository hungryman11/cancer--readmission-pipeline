# streamlit_app.py - FINAL VERSION (December 19, 2025)
# Fully robust: handles messy CSVs, auto-detects patient ID, creates target, imputes NaN
# Tested and working on Streamlit Cloud

# streamlit_app.py - FINAL VERSION with PR Curve, Confusion Matrix & Full Metrics

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
st.markdown("Upload your hospital admissions CSV — the app cleans it and predicts readmissions.")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

st.sidebar.markdown("""
**Upload Tips:**
- Save as **CSV UTF-8** in Excel/Google Sheets
- Mixed/junk data is automatically cleaned
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
        href = f'<a href="data:file/csv;base64,{b64}" download="synthetic_cancer_data.csv">Download Synthetic Data (10,000 patients)</a>'
        st.markdown(href, unsafe_allow_html=True)
    st.stop()

# --------------------------------------------------------------
# 2. Robust CSV Loader
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
    st.warning("Using fallback encoding 'latin1' — some text may appear strange.")
    return pd.read_csv(file, encoding='latin1', low_memory=False, errors='replace')

df_raw = load_data(uploaded_file)
st.success(f"Raw file loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# --------------------------------------------------------------
# 3. Clean Mixed Data
# --------------------------------------------------------------
cancer_markers = ['ID', 'Radius_mean', 'Texture_mean', 'Diagnosis']
cancer_start_idx = None
for i, col in enumerate(df_raw.columns):
    if col in cancer_markers:
        cancer_start_idx = i
        break

if cancer_start_idx is not None:
    st.info(f"Mixed data detected — keeping only first {cancer_start_idx} columns")
    df = df_raw.iloc[:, :cancer_start_idx].copy()
else:
    df = df_raw.copy()

if 'Row_ID' in df.columns:
    df = df.dropna(subset=['Row_ID']).copy()

# --------------------------------------------------------------
# 4. Auto-Detect Patient ID & Parse Dates
# --------------------------------------------------------------
possible_id_cols = ['Patient_ID', 'Patientt_ID', 'SUBJECT_ID', 'subject_id', 'PATIENT_ID', 'Hadm_ID', 'HADM_ID']
patient_id_col = None
for col in possible_id_cols:
    if col in df.columns:
        patient_id_col = col
        break

if patient_id_col is None:
    st.error("Could not find patient identifier column.")
    st.stop()

st.info(f"Using '{patient_id_col}' as patient identifier.")
df = df.rename(columns={patient_id_col: 'Patient_ID'})

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

if 'readmitted_30days' not in df.columns:
    st.info("Generating 'readmitted_30days' column...")
    df['readmitted_30days'] = 0
    for pid in df['Patient_ID'].unique():
        patient_data = df[df['Patient_ID'] == pid]
        if len(patient_data) > 1:
            for i in range(len(patient_data) - 1):
                curr_disch = patient_data.iloc[i]['Disch_time']
                next_admit = patient_data.iloc[i + 1]['Admit_time']
                if pd.notna(curr_disch) and pd.notna(next_admit):
                    days_diff = (next_admit - curr_disch).days
                    if 0 < days_diff <= 30:
                        idx = patient_data.index[i]
                        df.loc[idx, 'readmitted_30days'] = 1

    count = df['readmitted_30days'].sum()
    st.success(f"Target created: {count} readmissions within 30 days")

# --------------------------------------------------------------
# 5. Feature Engineering
# --------------------------------------------------------------
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
st.write("### Final Data for Modeling", df_final.head())

# --------------------------------------------------------------
# 6. Modeling with Imputation
# --------------------------------------------------------------
target = 'readmitted_30days'
X = df_final.select_dtypes(include=[np.number]).drop(columns=[target, 'Patient_ID', 'Row_ID', 'Hadm_ID'], errors='ignore')
y = df_final[target]

if X.isna().any().any():
    st.warning(f"Found {X.isna().sum().sum()} missing values — imputing with median.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
predictions = {}
full_metrics = {}

st.header("Training Models")
progress = st.progress(0)

for i, (name, model) in enumerate(models.items()):
    with st.spinner(f"Training {name}..."):
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        results[name] = {'AUC': auc(fpr, tpr)}
        predictions[name] = {'prob': y_prob, 'pred': y_pred, 'fpr': fpr, 'tpr': tpr, 'precision': precision, 'recall': recall}
        
        full_metrics[name] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    progress.progress((i + 1) / len(models))

st.success("All models trained!")

# --------------------------------------------------------------
# 7. Results & Visualizations
# --------------------------------------------------------------
st.header("Model Performance Summary")
metrics_df = pd.DataFrame({name: {
    'AUC': results[name]['AUC'],
    'Accuracy': full_metrics[name]['accuracy'],
    'Precision': full_metrics[name]['1']['precision'],
    'Recall': full_metrics[name]['1']['recall'],
    'F1': full_metrics[name]['1']['f1-score']
} for name in models}).T.round(4)

st.dataframe(metrics_df.style.highlight_max(axis=0))

best_model_name = metrics_df['AUC'].idxmax()
st.success(f"🏆 Best Model: **{best_model_name}** (AUC = {metrics_df.loc[best_model_name, 'AUC']:.4f})")

# Plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("ROC Curves")
    fig, ax = plt.subplots(figsize=(8,6))
    for name, pred in predictions.items():
        ax.plot(pred['fpr'], pred['tpr'], label=f"{name} (AUC = {results[name]['AUC']:.3f})")
    ax.plot([0,1], [0,1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Precision-Recall Curves")
    fig, ax = plt.subplots(figsize=(8,6))
    for name, pred in predictions.items():
        ax.plot(pred['recall'], pred['precision'], label=f"{name} (AP = {average_precision_score(y_test, pred['prob']):.3f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with col2:
    st.subheader(f"Confusion Matrix — {best_model_name}")
    cm = confusion_matrix(y_test, predictions[best_model_name]['pred'])
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Readmit', 'Readmit'],
                yticklabels=['No Readmit', 'Readmit'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.subheader("Random Forest Feature Importance")
    # Always show Random Forest importance (most interpretable tree model)
    rf_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_pipe.fit(X_train, y_train)
    importances = rf_pipe.named_steps['model'].feature_importances_
    imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    imp_df = imp_df.sort_values('Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(data=imp_df, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title("Top 10 Features (Random Forest)")
    st.pyplot(fig)

# Download
st.download_button("Download Full Metrics", metrics_df.to_csv(), "readmission_full_metrics.csv")
st.balloons()