import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
        df = pd.DataFrame(X, columns=[f'feature
