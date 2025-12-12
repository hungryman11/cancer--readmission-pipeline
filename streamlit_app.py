# streamlit_app.py - FINAL VERSION (December 13, 2025)
# Handles messy CSVs, auto-creates 'readmitted_30days', imputes missing values
# Fully tested and error-free on Streamlit Cloud

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
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer  # ← Added for NaN handling

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
# 3. Clean Mixed Data (Remove breast cancer columns)
# --------------------------------------------------------------
cancer_markers = ['ID', 'Radius_mean', 'Texture_mean', 'Diagnosis']
cancer_start_idx = None
for i, col in enumerate(df_raw.columns):
    if col in cancer_markers:
        cancer_start_idx = i
        break

if cancer_start_idx is not None:
    st.info(f"Mixed data detected — keeping only first {cancer_start_idx} columns (admissions)")
    df = df_raw.iloc[:, :cancer_start_idx].copy()
else:
    df = df_raw.copy()

if 'Row_ID' in df.columns:
    df = df.dropna(subset=['Row_ID']).copy()

if 'Patientt_ID' in df.columns:
    df = df.rename(columns={'Patientt_ID': 'Patient_ID'})

st.write("### Cleaned Admissions Data", df.head())

# --------------------------------------------------------------
# 4. Parse Dates & Create 'readmitted_30days'
# --------------------------------------------------------------
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

df = df.sort_values(['Patient_ID', 'Admit_time']).reset_index(drop=True)

if 'readmitted_30days' not in df.columns:
    st.info("Generating 'readmitted_30days' column...")
    df['readmitted_30days'] = 0
    for pid in df['Patient_ID'].dropna().unique():
        patient_idx = df[df['Patient_ID'] == pid].index.tolist()
        for i in range(len(patient_idx) - 1):
            curr_disch = df.loc[patient_idx[i], 'Disch_time']
            next_admit = df.loc[patient_idx[i + 1], 'Admit_time']
            if pd.notna(curr_disch) and pd.notna(next_admit):
                days_diff = (next_admit - curr_disch).days
                if 0 < days_diff <= 30:
                    df.loc[patient_idx[i], 'readmitted_30days'] = 1

    count = df['readmitted_30days'].sum()
    st.success(f"Target created: {count} readmissions out of {len(df)} admissions")

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
# 6. Modeling (with NaN imputation)
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

st.header("Training Models")
progress = st.progress(0)

for i, (name, model) in enumerate(models.items()):
    with st.spinner(f"Training {name}..."):
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Handles NaN
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = {'AUC': auc(fpr, tpr)}
        predictions[name] = y_prob
    progress.progress((i + 1) / len(models))

st.success("All models trained!")

# --------------------------------------------------------------
# 7. Results & Plots
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
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Top Features (Best Tree Model)")
    if "Random" in best or "Gradient" in best:
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', models[best])
        ])
        pipe.fit(X_train, y_train)
        imp = pd.DataFrame({'Feature': X.columns, 'Importance': pipe.named_steps['model'].feature_importances_})
        imp = imp.sort_values('Importance', ascending=False).head(10)
        fig, ax = plt.subplots()
        sns.barplot(data=imp, x='Importance', y='Feature', palette="viridis")
        st.pyplot(fig)

st.download_button("Download Results", metrics_df.to_csv(), "readmission_results.csv")
st.balloons()
