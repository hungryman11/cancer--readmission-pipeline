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

st.set_page_config(page_title="Cancer Readmission Predictor", layout="wide")
st.title("🏥 Cancer Patient 30-Day Readmission Prediction")
st.markdown("Upload a CSV or generate synthetic data for a demo.")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Synthetic data demo
if uploaded_file is None:
    st.info("No file uploaded – generate demo data below.")
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
        df.loc[high_risk, 'age'] = (df.loc[high_risk, 'age'] + np.random.normal(8, 5, high_risk.sum()).astype(int)).clip(18, 90)
        df.loc[high_risk, 'length_of_stay'] += np.random.poisson(4, high_risk.sum())

        csv = df.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="synthetic_cancer_data.csv">Download Synthetic CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    st.stop()

# Load data
@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        file.seek(0)
        return pd.read_csv(file, encoding='cp1252')
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

df = load_data(uploaded_file)
st.success(f"Loaded {df.shape[0]:,} rows")

# Feature engineering
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    if 'age' in df.columns:
        df['is_elderly'] = (df['age'] >= 65).astype(int)
    if 'cancer_stage' in df.columns:
        df['is_advanced_stage'] = df['cancer_stage'].str.contains('III|IV|3|4', case=False, na=False).astype(int)
    if 'chemo_cycles' in df.columns:
        df['has_chemotherapy'] = (df['chemo_cycles'] > 0).astype(int)
        df['intensive_chemo'] = (df['chemo_cycles'] >= 6).astype(int)
    if 'length_of_stay' in df.columns:
        df['prolonged_stay'] = (df['length_of_stay'] > 7).astype(int)
    if 'previous_admissions' in df.columns:
        df['frequent_admitter'] = (df['previous_admissions'] >= 3).astype(int)
    if 'emergency_admission' in df.columns:
        df['emergency_admission'] = df['emergency_admission'].astype(int)
    if 'icu_stay' in df.columns:
        df['had_icu_stay'] = df['icu_stay'].astype(int)
    if 'hemoglobin' in df.columns:
        df['anemia'] = (df['hemoglobin'] < 10).astype(int)
    if 'albumin' in df.columns:
        df['hypoalbuminemia'] = (df['albumin'] < 3.5).astype(int)
    if 'num_medications' in df.columns:
        df['polypharmacy'] = (df['num_medications'] >= 10).astype(int)

    # Fixed: Use exact column names
    risk_cols = ['is_elderly', 'is_advanced_stage', 'has_chemotherapy', 'intensive_chemo',
                 'prolonged_stay', 'frequent_admitter', 'emergency_admission', 'had_icu_stay',
                 'anemia', 'hypoalbuminemia', 'polypharmacy']
    risk_cols = [c for c in risk_cols if c in df.columns]
    if risk_cols:
        df['composite_risk_score'] = df[risk_cols].sum(axis=1)
        df['high_risk_patient'] = (df['composite_risk_score'] >= 2).astype(int)
    return df

df_eng = engineer_features(df)
st.write("### Feature Engineering Preview", df_eng.head(10))

# Continue with modeling, results, plots, etc. (add the full section from previous versions)

# ... [rest of the code: modeling, ROC, importance, download] ...
