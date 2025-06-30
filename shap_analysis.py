import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt # For SHAP plots that are hard to convert to Plotly directly

# Load the dataset, model and scaler
df = pd.read_csv("bank_customer_data.csv")
lgbm = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Prepare data for SHAP
X = df.drop(["customer_id", "month", "inadimplencia"], axis=1)
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Create a SHAP explainer
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_scaled)

# Fix for summary plot dimension issue
# shap_values[1] is for the positive class (default=1)
if isinstance(shap_values, list):
    shap_values_for_plot = shap_values[1]
else:
    shap_values_for_plot = shap_values

# 1. Summary Plot (Bar Plot)
fig_summary_bar = shap.summary_plot(shap_values_for_plot, X_scaled, plot_type="bar", show=False)
plotly_summary_bar = go.Figure(fig_summary_bar)
plotly_summary_bar.update_layout(title_text="SHAP Importância das Features (Gráfico de Barras)")
plotly_summary_bar.write_html("shap_summary_bar.html")

# 2. Summary Plot (Beeswarm Plot)
fig_beeswarm = shap.summary_plot(shap_values_for_plot, X_scaled, show=False)
plotly_beeswarm = go.Figure(fig_beeswarm)
plotly_beeswarm.update_layout(title_text="SHAP Summary Plot (Beeswarm Plot)")
plotly_beeswarm.write_html("shap_beeswarm.html")

# 3. Dependence Plot for a few key features (to avoid too many files)
# We will select top 5 features for static plots, and allow selection in Streamlit
top_features = X_scaled.columns[np.argsort(np.sum(np.abs(shap_values_for_plot), axis=0))[::-1][:5]]
for feature in top_features:
    fig_dependence = shap.dependence_plot(feature, shap_values_for_plot, X_scaled, show=False)
    plotly_dependence = go.Figure(fig_dependence)
    plotly_dependence.update_layout(title_text=f"SHAP Dependence Plot para {feature}")
    plotly_dependence.write_html(f"shap_dependence_{feature}.html")

# Force Plot, Decision Plot, Waterfall Plot, Heatmap
# These are best rendered dynamically in Streamlit due to their interactive nature or complexity.
# For static files, they often require converting matplotlib figures to images.
# We will implement them directly in Streamlit.

print("SHAP analysis completed and plots saved as HTML.")


