"""
app.py
------
Streamlit dashboard for FairML.
"""

import streamlit as st
import pandas as pd

from models.example_model import load_model, train_example_model
from fairness_tests.fairness import demographic_parity_difference
from fairness_tests.bias import introduce_bias
from dashboard.components.charts import plot_fairness_bar
