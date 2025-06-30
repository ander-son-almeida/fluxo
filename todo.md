## Tarefas

### Fase 1: Expandir dataset bancário para 50+ features
- [x] Gerar dataset sintético com mais de 50 features mensais e coluna 'inadimplencia' em português

### Fase 2: Retreinar modelo LightGBM com novo dataset
- [x] Carregar e pré-processar os dados
- [x] Dividir dados em treino e teste
- [x] Aplicar técnicas de tratamento de desbalanceamento (e.g., SMOTE, undersampling)
- [x] Treinar modelo LightGBM
- [ ] Otimizar hiperparâmetros (opcional)

### Fase 3: Gerar todos os gráficos SHAP em Plotly
- [x] Calcular valores SHAP
- [x] Gerar gráficos SHAP (Summary Plot, Bar Plot, Force Plot, Decision Plot, Dependence Plot, Waterfall Plot, Beeswarm Plot, Heatmap) em Plotly

### Fase 4: Gerar todos os gráficos de avaliação do modelo em Plotly
- [x] Gerar gráficos de importância de features
- [x] Gerar curva ROC
- [x] Gerar curva KS
- [x] Gerar curva de calibração
- [x] Gerar curva Precision-Recall
- [x] Gerar matriz de confusão

### Fase 5: Atualizar aplicação Streamlit com novos gráficos e funcionalidades
- [x] Estruturar o aplicativo Streamlit
- [x] Integrar o modelo treinado
- [x] Adicionar campo para seleção de cliente e visualização SHAP
- [x] Exibir gráficos de avaliação do modelo

### Fase 6: Testar e entregar aplicação completa
- [x] Testar todas as funcionalidades do Streamlit
- [x] Preparar para entrega

