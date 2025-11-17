# dashboard/app.py
import streamlit as st
import pandas as pd

from src.models.example_model import load_model, train_example_model
from src.fairness_tests.fairness import demographic_parity_difference
from src.fairness_tests.bias import introduce_bias
from dashboard.components.charts import plot_fairness_bar

st.set_page_config(page_title="FairML Dashboard", layout="centered")

st.title("FairML Dashboard")

df = pd.DataFrame({
    "feature": ["A", "B", "C"],
    "value": [0.2, 0.5, 0.8],
})

model = load_model()
if model is None:
    model = train_example_model()

bias_df = introduce_bias(df, "feature")
fairness_score = demographic_parity_difference(bias_df, "feature", "value")

st.write("### Biased Data")
st.dataframe(bias_df)

st.write("### Fairness Score")
st.write(fairness_score)

st.write("### Fairness Visualization")
plot_fairness_bar(fairness_score)
