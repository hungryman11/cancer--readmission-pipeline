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
