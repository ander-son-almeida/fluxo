import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
import plotly.express as px

# Load the model, X_test, and y_test
lgbm = joblib.load("lightgbm_model.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

# Predictions
y_pred = lgbm.predict(X_test)
y_pred_proba = lgbm.predict_proba(X_test)[:, 1]

# 1. Feature Importance
feature_importance = pd.DataFrame({
    'feature': X_test.columns,
    'importance': lgbm.feature_importances_
}).sort_values(by='importance', ascending=False)

fig_feature_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                                title='Importância das Features')
fig_feature_importance.write_html("feature_importance.html")

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Curva ROC (AUC = {roc_auc:.3f})'))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Classificador Aleatório', line=dict(dash='dash')))
fig_roc.update_layout(title='Curva ROC',
                      xaxis_title='Taxa de Falsos Positivos',
                      yaxis_title='Taxa de Verdadeiros Positivos')
fig_roc.write_html("roc_curve.html")

# 3. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Curva Precision-Recall'))
fig_pr.update_layout(title='Curva Precision-Recall',
                     xaxis_title='Recall',
                     yaxis_title='Precision')
fig_pr.write_html("precision_recall_curve.html")

# 4. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                  labels=dict(x="Predito", y="Real", color="Contagem"),
                  x=['Bom Pagador', 'Inadimplente'], y=['Bom Pagador', 'Inadimplente'],
                  title="Matriz de Confusão")
fig_cm.write_html("confusion_matrix.html")

# 5. Calibration Curve (Reliability Diagram)
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)

fig_cal = go.Figure()
fig_cal.add_trace(go.Scatter(x=mean_predicted_value, y=fraction_of_positives, mode='lines+markers', name='Calibração do Modelo'))
fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfeitamente Calibrado', line=dict(dash='dash')))
fig_cal.update_layout(title='Curva de Calibração',
                      xaxis_title='Probabilidade Média Predita',
                      yaxis_title='Fração de Positivos')
fig_cal.write_html("calibration_curve.html")

# 6. KS Curve (Kolmogorov-Smirnov)
df_results = pd.DataFrame({'true_label': y_test, 'predicted_proba': y_pred_proba})
df_results = df_results.sort_values('predicted_proba', ascending=False)

# Calculate cumulative sums for good and bad customers
total_good = df_results['true_label'].value_counts()[0]
total_bad = df_results['true_label'].value_counts()[1]

df_results['cumulative_good'] = df_results['true_label'].apply(lambda x: 1 if x == 0 else 0).cumsum()
df_results['cumulative_bad'] = df_results['true_label'].apply(lambda x: 1 if x == 1 else 0).cumsum()

df_results['percent_good'] = df_results['cumulative_good'] / total_good
df_results['percent_bad'] = df_results['cumulative_bad'] / total_bad

ks_statistic = np.max(np.abs(df_results['percent_good'] - df_results['percent_bad']))

fig_ks = go.Figure()
fig_ks.add_trace(go.Scatter(x=np.arange(len(df_results)), y=df_results['percent_good'], mode='lines', name='Clientes Bons Acumulados'))
fig_ks.add_trace(go.Scatter(x=np.arange(len(df_results)), y=df_results['percent_bad'], mode='lines', name='Clientes Inadimplentes Acumulados'))
fig_ks.update_layout(title=f'Curva KS (Estatística KS: {ks_statistic:.3f})',
                     xaxis_title='Número de Amostras',
                     yaxis_title='Porcentagem Acumulada')
fig_ks.write_html("ks_curve.html")

print("Model evaluation plots generated successfully!")


