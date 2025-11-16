"""
app.py
------
Streamlit dashboard for FairML.
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
from src.models.example_model import load_model, train_example_model
from src.fairness_tests.fairness import demographic_parity_difference
from src.fairness_tests.bias import introduce_bias
from dashboard.components.charts import plot_fairness_bar

st.set_page_config(page_title="FairML Dashboard", layout="centered")

st.title("FairML – Model Fairness Dashboard")
st.markdown("Evaluate model fairness and bias across protected groups.")

# Load or train model
try:
    model = load_model()
    st.success("Model loaded successfully!")
except:
    st.info("Training example model...")
    model = train_example_model()
    st.success("Model trained and loaded!")

# Load data
data = pd.read_csv('src/data/synthetic_data.csv')
protected_feature = 'feature_0'

# Bias introduction slider
bias_level = st.slider("Introduce Bias Level", 0.0, 0.5, 0.1, 0.01)
biased_data = introduce_bias(data, protected_feature, bias_level)
X = biased_data.drop('target', axis=1)
y = biased_data['target']

# Evaluate fairness
dp_diff = demographic_parity_difference(model, X, y, protected_feature)
st.write(f"Demographic Parity Difference: {dp_diff:.2f}")

# Plot fairness chart
metric_dict = {f'Group {g}': dp_diff for g in X[protected_feature].unique()}
chart = plot_fairness_bar(metric_dict)
st.altair_chart(chart, use_container_width=True)

# Raw data expander
with st.expander("Show Raw Data"):
    st.dataframe(data)
