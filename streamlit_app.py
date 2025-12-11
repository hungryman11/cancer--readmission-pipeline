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
st.markdown("Upload a CSV or generate synthetic data for a demo.")

# --------------------------------------------------------------
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

st.sidebar.markdown("""
**Tip:** Having trouble uploading?
- Open your file in Excel/Google Sheets
- Save as **CSV UTF-8 (Comma delimited)**
- Do not upload Excel (.xlsx) files
""")

# --------------------------------------------------------------
# 1. Generate Synthetic Data (Demo Mode)
# --------------------------------------------------------------
if uploaded_file is None:
    st.info("No file uploaded yet — generate demo data below.")
    
    if st.button("Generate & Download Synthetic Demo Data"):
        X, y = make_classification(
            n_samples=10000,
            n_features=20,
            weights=[0.85, 0.15],
            random_state=42
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['readmitted_30days'] = y

        np.random.seed(42)
        df['patient_id'] = range(1, len(df) + 1)
        df['age'] = np.random.normal(65, 12, len(df)).clip(18, 90).astype(int)
        df['cancer_stage'] = np.random.choice(
            ['Stage I', 'Stage II', 'Stage III', 'Stage IV'],
            size=len(df), p=[0.25, 0.35, 0.25, 0.15]
        )
        df['chemo_cycles'] = np.random.poisson(4, len(df))
        df['length_of_stay'] = np.random.lognormal(1.8, 0.7, len(df)).astype(int) + 1
        df['num_medications'] = np.random.poisson(8, len(df)) + 1
        df['previous_admissions'] = np.random.poisson(1.5, len(df))
        df['icu_stay'] = np.random.binomial(1, 0.2, len(df))
        df['emergency_admission'] = np.random.binomial(1, 0.4, len(df))
        df['hemoglobin'] = np.random.normal(12, 2, len(df)).clip(6, 18)
        df['albumin'] = np.random.normal(4.0, 0.6, len(df)).clip(2, 5.5)

        # Make readmitted patients look sicker
        high_risk = df['readmitted_30days'] == 1
        df.loc[high_risk, 'age'] = (df.loc[high_risk, 'age'] + np.random.normal(8, 5, high_risk.sum())).clip(18, 95).astype(int)
        df.loc[high_risk, 'length_of_stay'] += np.random.poisson(5, high_risk.sum())

        csv = df.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="synthetic_cancer_data.csv">Download Synthetic CSV (10,000 patients)</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    st.stop()

# --------------------------------------------------------------
# 2. Ultra-Robust CSV Loader
# --------------------------------------------------------------
@st.cache_data
def load_data(file):
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1', 'iso-8859-1']
    for enc in encodings:
        try:
            file.seek(0)
            df = pd.read_csv(file, encoding=enc)
            st.success(f"Loaded {df.shape[0]:,} rows using encoding: **{enc}**")
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError as e:
            st.error(f"CSV format error: {e}")
            st.stop()
        except Exception:
            continue
    
    # Final fallback
    st.warning("Trying last-resort encoding 'latin1'...")
    file.seek(0)
    try:
        df = pd.read_csv(file, encoding='latin1')
        st.success("Loaded with 'latin1' (some text may appear garbled)")
        return df
    except Exception as e:
        st.error("Could not read file with any encoding.")
        st.info("Please save your file as **CSV UTF-8** and try again.")
        st.stop()

df = load_data(uploaded_file)

# --------------------------------------------------------------
# 3. Feature Engineering
# --------------------------------------------------------------
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

    # Composite risk score — exact column names
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

# --------------------------------------------------------------
# 4. Modeling
# --------------------------------------------------------------
target = 'readmitted_30days'
if target not in df_eng.columns:
    st.error(f"Column '{target}' not found! Your CSV must contain this target column.")
    st.stop()

# Use only numeric features
feature_cols = df_eng.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in feature_cols if c not in [target, 'patient_id']]

if len(feature_cols) == 0:
    st.error("No numeric features found for modeling!")
    st.stop()

X = df_eng[feature_cols]
y = df_eng[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}
predictions = {}

st.header("Training Models...")
progress = st.progress(0)

for i, (name, model) in enumerate(models.items()):
    st.write(f"Training **{name}**...")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    results[name] = {
        'AUC': auc(fpr, tpr),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0)
    }
    predictions[name] = y_prob
    progress.progress((i+1)/len(models))

st.success("All models trained!")

# --------------------------------------------------------------
# 5. Results
# --------------------------------------------------------------
st.header("Model Performance")
metrics_df = pd.DataFrame(results).T.round(4)
st.dataframe(metrics_df.style.highlight_max(axis=0))

best_model = metrics_df['AUC'].idxmax()
st.success(f"Best Model: **{best_model}** – AUC = {metrics_df.loc[best_model, 'AUC']:.4f}")

# ROC + Feature Importance
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
    st.subheader("Top 10 Features (Best Tree Model)")
    if "Random" in best_model or "Gradient" in best_model:
        pipe = Pipeline([('scaler', StandardScaler()), ('model', models[best_model])])
        pipe.fit(X_train, y_train)
        imp = pipe.named_steps['model'].feature_importances_
        imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': imp})
        imp_df = imp_df.sort_values('Importance', ascending=False).head(10)
        
        fig, ax = plt.subplots()
        sns.barplot(data=imp_df, x='Importance', y='Feature', palette="viridis")
        st.pyplot(fig)
    else:
        st.info("Logistic Regression has no feature importance.")

# Download results
csv = metrics_df.to_csv().encode()
st.download_button("Download Model Results", csv, "cancer_readmission_results.csv", "text/csv")

st.balloons()
