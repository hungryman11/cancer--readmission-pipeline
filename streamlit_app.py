import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import time
import base64

# Page config
st.set_page_config(page_title="Cancer Readmission Predictor", layout="wide")
st.title("🏥 Cancer Patient 30-Day Readmission Prediction System")
st.markdown("Upload your patient data (CSV) and get instant ML predictions with full visualizations.")

# Sidebar
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

if uploaded_file is None:
    st.info("👈 Please upload a CSV file to start. For demo, generate synthetic data below.")
    
    if st.button("Generate & Download Synthetic Demo Data"):
        from sklearn.datasets import make_classification
        
        # Generate base synthetic data
        X, y = make_classification(n_samples=10000, n_features=20, weights=[0.85, 0.15], random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])  # ← Fixed f-string
        df['readmitted_30days'] = y
        
        # Add realistic clinical columns
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
        
        # Boost risk for readmitted patients
        high_risk = df['readmitted_30days'] == 1
        df.loc[high_risk, 'age'] += np.random.normal(8, 5, high_risk.sum()).clip(-20, 20).astype(int)
        df.loc[high_risk, 'length_of_stay'] += np.random.poisson(4, high_risk.sum())
        
        csv = df.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="synthetic_cancer_data.csv">Download Synthetic Data CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    st.stop()

# Load data
@st.cache_data
def load_data(file):
    try:
        # First try the standard UTF-8
        return pd.read_csv(file)
    except UnicodeDecodeError:
        # If that fails, rewind and try common alternative encodings
        file.seek(0)
        try:
            return pd.read_csv(file, encoding='cp1252')  # Windows common
        except UnicodeDecodeError:
            file.seek(0)
            try:
                return pd.read_csv(file, encoding='latin1')  # Very permissive
            except UnicodeDecodeError:
                file.seek(0)
                return pd.read_csv(file, encoding='iso-8859-1')  # Fallback
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()
        )

df = load_data(uploaded_file)
st.success(f"✅ Loaded {len(df):,} patient records with {len(df.columns)} columns")

# Feature Engineering
def engineer_features(df):
    df = df.copy()
    
    # Demographics
    if 'age' in df.columns:
        df['is_elderly'] = (df['age'] >= 65).astype(int)
    
    # Cancer stage
    if 'cancer_stage' in df.columns:
        df['is_advanced_stage'] = df['cancer_stage'].str.contains('III|IV|3|4', case=False, na=False).astype(int)
    
    # Treatment
    if 'chemo_cycles' in df.columns:
        df['has_chemotherapy'] = (df['chemo_cycles'] > 0).astype(int)
        df['intensive_chemo'] = (df['chemo_cycles'] >= 6).astype(int)
    
    # Admission patterns
    if 'length_of_stay' in df.columns:
        df
