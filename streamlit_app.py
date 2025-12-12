# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import time

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------
st.set_page_config(page_title="Cancer Readmission Predictor", layout="wide")
st.title("Cancer Patient 30-Day Readmission Prediction")
st.markdown("Upload your CSV or generate synthetic data below.")

# --------------------------------------------------------------
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

st.sidebar.markdown("""
**Having upload issues?**
- Open in Excel → Save As → **CSV UTF-8**
- Avoid Excel (.xlsx) files
""")

# --------------------------------------------------------------
# 1. Synthetic Data Demo
# --------------------------------------------------------------
if uploaded_file is None:
    st.info("No file uploaded — generate demo data below.")
    if st.button("Generate & Download Synthetic Demo Data"):
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
# 2. BULLETPROOF CSV LOADER — This fixes ALL encoding errors
# --------------------------------------------------------------
@st.cache_data
def load_data(file):
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'iso-8859-1', 'latin1']
    for enc in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except:
            continue
    # Ultimate fallback — latin1 never fails
    file.seek(0)
    st.warning("Using fallback encoding (latin1) — file loaded but some text may look strange.")
    return pd.read_csv(file, encoding='latin1', errors='replace')

df = load_data(uploaded_file)
st.success(f"Loaded {df.shape[0]:,} patients successfully!")

# --------------------------------------------------------------
# 3. Feature Engineering (Safe & Clean)
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

    risk_cols = ['is_elderly', 'is_advanced_stage', 'has_chemotherapy',
                 'prolonged_stay', 'polypharmacy', 'anemia', 'hypoalbuminemia']
    risk_cols = [c for c in risk_cols if c in df.columns]
    if risk_cols:
        df['composite_risk_score'] = df[risk_cols].sum(axis=1)
        df['high_risk_patient'] = (df['composite_risk_score'] >= 2).astype(int)
    return df

df_eng = engineer_features(df)
st.write("### Engineered Features", df_eng.head(10))

# --------------------------------------------------------------
# 4. Modeling
# --------------------------------------------------------------
target = 'readmitted_30days'
if target not in df_eng.columns:
    st.error(f"Column '{target}' not found! CSV must contain this target.")
    st.stop()

X = df_eng.select_dtypes(include=[np.number]).drop(columns=[target, 'patient_id'], errors='ignore')
y = df_eng[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
predictions = {}
st.header("Training Models...")
for name, model in models.items():
    with st.spinner(f"Training {name}..."):
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        results[name] = {'AUC': auc(fpr, tpr)}
        predictions[name] = prob

st.success("Training complete!")

# --------------------------------------------------------------
# 5. Results
# --------------------------------------------------------------
st.header("Model Performance")
metrics_df = pd.DataFrame(results).T.round(3)
st.dataframe(metrics_df.style.highlight_max(axis=0))

best = metrics_df['AUC'].idxmax()
st.success(f"Best Model: **{best}** – AUC = {metrics_df.loc[best, 'AUC']:.3f}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("ROC Curves")
    fig, ax = plt.subplots()
    for name, prob in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} ({results[name]['AUC']:.3f})")
    ax.plot([0,1],[0,1],'k--')
    ax.legend()
    st.pyplot(fig)

with col2:
    if "Random" in best or "Gradient" in best:
        pipe = Pipeline([('scaler', StandardScaler()), ('model', models[best])])
        pipe.fit(X_train, y_train)
        imp = pd.DataFrame({
            'Feature': X.columns,
            'Importance': pipe.named_steps['model'].feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        fig, ax = plt.subplots()
        sns.barplot(data=imp, x='Importance', y='Feature', ax=ax)
        st.pyplot(fig)

st.download_button("Download Results", metrics_df.to_csv(), "results.csv")

st.balloons()

# After df = load_data(uploaded_file)

# Detect where cancer data starts (column 'ID' or 'Radius_mean' etc.)
cancer_start_idx = None
for i, col in enumerate(df.columns):
    if col in ['ID', 'Diagnosis', 'Radius_mean']:  # Markers for breast cancer data
        cancer_start_idx = i
        break

if cancer_start_idx is not None:
    st.info(f"Detected mixed data — keeping only admissions columns (first {cancer_start_idx})")
    df = df.iloc[:, :cancer_start_idx]

# Drop rows without Row_ID (cancer rows)
df = df.dropna(subset=['Row_ID']).copy()

# Fix column name typo
if 'Patientt_ID' in df.columns:
    df = df.rename(columns={'Patientt_ID': 'Patient_ID'})

# Parse Admit_time and Disch_time
def parse_datetime(date_str, year_col, default_time='00:00'):
    if pd.isna(date_str) or pd.isna(year_col):
        return pd.NaT
    parts = str(date_str).split(' ')
    date_part = parts[0]
    time_part = parts[1] if len(parts) > 1 else default_time
    try:
        mm, dd, yy = date_part.split('/')
        full_year = int(year_col)
        dt_str = f"{mm.zfill(2)}/{dd.zfill(2)}/{full_year} {time_part}"
        return pd.to_datetime(dt_str, format='%m/%d/%Y %H:%M', errors='coerce')
    except:
        return pd.NaT

df['Admit_time'] = df.apply(lambda row: parse_datetime(row['Admitted_date'], row['Admitted_year'], '00:00'), axis=1)
df['Disch_time'] = df.apply(lambda row: parse_datetime(row['Disch_date'], row['Disch_year'], '00:00'), axis=1)

# Sort for readmission logic
df = df.sort_values(['Patient_ID', 'Admit_time']).reset_index(drop=True)

# Add readmitted_30days column
df['readmitted_30days'] = 0

for patient_id in df['Patient_ID'].unique():
    patient_rows = df[df['Patient_ID'] == patient_id].index
    for i in range(len(patient_rows) - 1):
        current_idx = patient_rows[i]
        next_idx = patient_rows[i + 1]
        disch = df.loc[current_idx, 'Disch_time']
        next_admit = df.loc[next_idx, 'Admit_time']
        if pd.notna(disch) and pd.notna(next_admit):
            days_diff = (next_admit - disch).days
            if 0 < days_diff <= 30:
                df.loc[current_idx, 'readmitted_30days'] = 1

st.success(f"Cleaned data: {df.shape[0]} admissions, {df['readmitted_30days'].sum()} readmissions within 30 days")
st.write("### Cleaned Data Preview", df.head(10))

# Now continue with feature engineering and modeling on this cleaned df
df_eng = engineer_features(df)  # Use your existing function
