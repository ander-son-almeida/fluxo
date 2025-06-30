import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import io
import base64

# Set page config
st.set_page_config(page_title="Análise de Inadimplência Bancária", layout="wide")

# Load models and data
@st.cache_data
def load_data():
    df = pd.read_csv("bank_customer_data.csv")
    lgbm = joblib.load("lightgbm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
    return df, lgbm, scaler, X_test, y_test

# Load SHAP explainer
@st.cache_resource
def load_shap_explainer(_lgbm, X_test):
    explainer = shap.TreeExplainer(_lgbm)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values

# Convert matplotlib figure to base64 for display
def fig_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format=\'png\', bbox_inches=\'tight\')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Main app
def main():
    st.title("🏦 Análise de Inadimplência Bancária com LightGBM e SHAP")
    
    # Load data
    df, lgbm, scaler, X_test, y_test = load_data()
    explainer, shap_values = load_shap_explainer(lgbm, X_test)
    
    # Sidebar
    st.sidebar.title("Navegação")
    page = st.sidebar.selectbox("Escolha uma página:", 
                               ["Visão Geral", "Avaliação do Modelo", "Análise SHAP", "Predição Individual"])
    
    if page == "Visão Geral":
        show_overview(df)
    elif page == "Avaliação do Modelo":
        show_model_evaluation(lgbm, X_test, y_test)
    elif page == "Análise SHAP":
        show_shap_analysis(explainer, shap_values, X_test)
    elif page == "Predição Individual":
        show_individual_prediction(df, lgbm, scaler, explainer, X_test)

def show_overview(df):
    st.header("📊 Visão Geral dos Dados")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Clientes", df[\'customer_id\'].nunique())
    
    with col2:
        st.metric("Total de Registros", len(df))
    
    with col3:
        default_rate = df[\'inadimplencia\'].mean() * 100
        st.metric("Taxa de Inadimplência", f"{default_rate:.2f}%")
    
    with col4:
        st.metric("Meses de Dados", df[\'month\'].nunique())
    
    st.subheader("Distribuição das Features Principais")
    
    # Feature distribution plots for main features
    main_features = [\'pontuacao_credito\', \'renda\', \'valor_emprestimo\', \'historico_pagamento\', \'anos_emprego\']
    
    for feature in main_features:
        fig = px.histogram(df, x=feature, color=\'inadimplencia\', 
                          title=f\'Distribuição de {feature} por Status de Inadimplência\',
                          marginal="box")
        st.plotly_chart(fig, use_container_width=True)

def show_model_evaluation(lgbm, X_test, y_test):
    st.header("📈 Avaliação do Modelo")
    
    # Predictions
    y_pred = lgbm.predict(X_test)
    y_pred_proba = lgbm.predict_proba(X_test)[:, 1]
    
    # Feature Importance
    st.subheader("Importância das Features")
    feature_importance = pd.DataFrame({
        \'feature\': X_test.columns,
        \'importance\': lgbm.feature_importances_
    }).sort_values(by=\'importance\', ascending=False).head(20)  # Show top 20
    
    fig_importance = px.bar(feature_importance, x=\'importance\', y=\'feature\', orientation=\'h\',
                           title=\'Top 20 Features - Importância\')
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # ROC Curve
    st.subheader("Curva ROC")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode=\'lines\', name=f\'Curva ROC (AUC = {roc_auc:.3f})\'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode=\'lines\', name=\'Classificador Aleatório\', line=dict(dash=\'dash\')))
    fig_roc.update_layout(title=\'Curva ROC\',
                         xaxis_title=\'Taxa de Falsos Positivos\',
                         yaxis_title=\'Taxa de Verdadeiros Positivos\')
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Precision-Recall Curve
    st.subheader("Curva Precision-Recall")
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode=\'lines\', name=\'Curva Precision-Recall\'))
    fig_pr.update_layout(title=\'Curva Precision-Recall\',
                        xaxis_title=\'Recall\',
                        yaxis_title=\'Precision\')
    st.plotly_chart(fig_pr, use_container_width=True)
    
    # Confusion Matrix
    st.subheader("Matriz de Confusão")
    cm = confusion_matrix(y_test, y_pred)
    
    fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                      title="Matriz de Confusão",
                      labels=dict(x="Predito", y="Real", color="Contagem"),
                      x=[\'Bom Pagador\', \'Inadimplente\'], y=[\'Bom Pagador\', \'Inadimplente\'])
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Calibration Curve
    st.subheader("Curva de Calibração")
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)
    
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=mean_predicted_value, y=fraction_of_positives, mode=\'lines+markers\', name=\'Calibração do Modelo\'))
    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode=\'lines\', name=\'Perfeitamente Calibrado\', line=dict(dash=\'dash\')))
    fig_cal.update_layout(title=\'Curva de Calibração\',
                         xaxis_title=\'Probabilidade Média Predita\',
                         yaxis_title=\'Fração de Positivos\')
    st.plotly_chart(fig_cal, use_container_width=True)
    
    # KS Curve
    st.subheader("Curva KS (Kolmogorov-Smirnov)")
    df_results = pd.DataFrame({\'true_label\': y_test, \'predicted_proba\': y_pred_proba})
    df_results = df_results.sort_values(\'predicted_proba\', ascending=False)
    
    total_good = df_results[\'true_label\'].value_counts()[0]
    total_bad = df_results[\'true_label\'].value_counts()[1]
    
    df_results[\'cumulative_good\'] = df_results[\'true_label\'].apply(lambda x: 1 if x == 0 else 0).cumsum()
    df_results[\'cumulative_bad\'] = df_results[\'true_label\'].apply(lambda x: 1 if x == 1 else 0).cumsum()
    
    df_results[\'percent_good\'] = df_results[\'cumulative_good\'] / total_good
    df_results[\'percent_bad\'] = df_results[\'cumulative_bad\'] / total_bad
    
    ks_statistic = np.max(np.abs(df_results[\'percent_good\'] - df_results[\'percent_bad\']))
    
    fig_ks = go.Figure()
    fig_ks.add_trace(go.Scatter(x=np.arange(len(df_results)), y=df_results[\'percent_good\'], mode=\'lines\', name=\'Clientes Bons Acumulados\'))
    fig_ks.add_trace(go.Scatter(x=np.arange(len(df_results)), y=df_results[\'percent_bad\'], mode=\'lines\', name=\'Clientes Inadimplentes Acumulados\'))
    fig_ks.update_layout(title=f\'Curva KS (Estatística KS: {ks_statistic:.3f})\',
                        xaxis_title=\'Número de Amostras\',
                        yaxis_title=\'Porcentagem Acumulada\')
    st.plotly_chart(fig_ks, use_container_width=True)

def show_shap_analysis(explainer, shap_values, X_test):
    st.header("🔍 Análise SHAP")
    
    # Fix for SHAP values format
    if isinstance(shap_values, list):
        shap_values_for_plot = shap_values[1]
    else:
        shap_values_for_plot = shap_values
    
    # 1. Summary Plot (Bar Plot)
    st.subheader("Importância Global das Features (SHAP)")
    
    # Create SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_for_plot, X_test, plot_type="bar", show=False)
    st.pyplot(plt.gcf())
    plt.close()
    
    # 2. SHAP Summary Plot (Beeswarm)
    st.subheader("SHAP Summary Plot (Beeswarm)")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_for_plot, X_test, show=False)
    st.pyplot(plt.gcf())
    plt.close()
    
    # 3. Dependence Plots
    st.subheader("SHAP Dependence Plots")
    feature_for_dependence = st.selectbox("Selecione uma feature para o Dependence Plot:", X_test.columns)
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_for_dependence, shap_values_for_plot, X_test, show=False)
    st.pyplot(plt.gcf())
    plt.close()
    
    # 4. Heatmap Plot
    st.subheader("SHAP Heatmap")
    # Select a subset of samples for the heatmap to avoid overcrowding
    sample_size = min(100, len(X_test))
    sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
    
    plt.figure(figsize=(12, 8))
    expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    shap.plots.heatmap(shap.Explanation(values=shap_values_for_plot[sample_indices], 
                                       base_values=expected_val, 
                                       data=X_test.iloc[sample_indices].values,
                                       feature_names=X_test.columns), show=False)
    st.pyplot(plt.gcf())
    plt.close()

def show_individual_prediction(df, lgbm, scaler, explainer, X_test):
    st.header("🎯 Predição Individual")
    
    # Customer selection
    customer_ids = df[\'customer_id\'].unique()
    selected_customer = st.selectbox("Selecione um cliente:", customer_ids)
    
    # Get customer data
    customer_data = df[df[\'customer_id\'] == selected_customer].iloc[-1]  # Get latest record
    
    # Display customer info
    st.subheader(f"Informações do Cliente {selected_customer}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Pontuação de Crédito", customer_data[\'pontuacao_credito\'])
        st.metric("Renda", f"${customer_data[\'renda\']:.2f}")
        st.metric("Valor do Empréstimo", f"${customer_data[\'valor_emprestimo\']:.2f}")
    
    with col2:
        st.metric("Histórico de Pagamento", customer_data[\'historico_pagamento\'])
        st.metric("Anos de Emprego", customer_data[\'anos_emprego\'])
        st.metric("Status Real", "Inadimplente" if customer_data[\'inadimplencia\'] == 1 else "Bom Pagador")
    
    # Prepare data for prediction
    features = X_test.columns
    customer_features = customer_data[features].values.reshape(1, -1)
    customer_features_scaled = scaler.transform(customer_features)
    
    # Make prediction
    prediction_proba = lgbm.predict_proba(customer_features_scaled)[0, 1]
    prediction = lgbm.predict(customer_features_scaled)[0]
    
    st.subheader("Predição do Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Probabilidade de Inadimplência", f"{prediction_proba:.3f}")
    
    with col2:
        st.metric("Predição", "Inadimplente" if prediction == 1 else "Bom Pagador")
    
    # SHAP explanation for this customer
    st.subheader("Explicação SHAP para este Cliente")
    
    customer_features_df = pd.DataFrame(customer_features_scaled, columns=features)
    shap_values_customer = explainer.shap_values(customer_features_df)
    
    if isinstance(shap_values_customer, list):
        shap_values_customer = shap_values_customer[1]
    
    # Force plot
    st.subheader("SHAP Force Plot")
    plt.figure(figsize=(12, 3))
    expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    shap.force_plot(expected_val, shap_values_customer[0], customer_features_df.iloc[0], 
                   matplotlib=True, show=False)
    st.pyplot(plt.gcf())
    plt.close()
    
    # Waterfall plot
    st.subheader("SHAP Waterfall Plot")
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(values=shap_values_customer[0], 
                                        base_values=expected_val, 
                                        data=customer_features_df.iloc[0].values,
                                        feature_names=features), show=False)
    st.pyplot(plt.gcf())
    plt.close()
    
    # Decision plot
    st.subheader("SHAP Decision Plot")
    plt.figure(figsize=(10, 6))
    shap.decision_plot(expected_val, shap_values_customer[0], customer_features_df.iloc[0], 
                      feature_names=features, show=False)
    st.pyplot(plt.gcf())
    plt.close()

if __name__ == "__main__":
    main()

