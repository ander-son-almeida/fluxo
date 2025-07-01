# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:24:23 2025

"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, chi2_contingency

# Funções Auxiliares

def get_leaf_intervals(tree, feature_names, class_names, node_index=0, current_rules=None):
    """ 
    Função recursiva para extrair os intervalos de decisão de cada folha da árvore.
    Percorre a árvore de decisão e, para cada folha, armazena as regras que levaram até ela.
    Inclui a impureza Gini da folha.
    """
    if current_rules is None:
        current_rules = []

    feature = tree.feature[node_index]
    threshold = tree.threshold[node_index]
    
    # Se for um nó folha, retorna as regras acumuladas, a classe predita e o Gini
    if feature == _tree.TREE_UNDEFINED:
        predicted_class_index = np.argmax(tree.value[node_index])
        predicted_class_name = class_names[predicted_class_index]
        gini_impurity = tree.impurity[node_index]
        return [(current_rules, predicted_class_name, gini_impurity)]

    feature_name = feature_names[feature]
    intervals = []

    # Regra para o filho da esquerda (feature <= threshold)
    left_rules = current_rules + [(feature_name, "<=", threshold)]
    intervals.extend(get_leaf_intervals(tree, feature_names, class_names, tree.children_left[node_index], left_rules))

    # Regra para o filho da direita (feature > threshold)
    right_rules = current_rules + [(feature_name, ">", threshold)]
    intervals.extend(get_leaf_intervals(tree, feature_names, class_names, tree.children_right[node_index], right_rules))

    return intervals

def format_intervals(leaf_intervals, df_types):
    """
    Formata os intervalos extraídos, combinando regras para a mesma feature e ajustando para tipos inteiros.
    """
    formatted = []
    for rules, predicted_class, gini_impurity in leaf_intervals:
        rule_dict = {}
        for feature, op, value in rules:
            if feature not in rule_dict:
                rule_dict[feature] = {"min": -np.inf, "max": np.inf}
            
            if op == "<=":
                rule_dict[feature]["max"] = min(rule_dict[feature]["max"], value)
            else: # op == ">"
                rule_dict[feature]["min"] = max(rule_dict[feature]["min"], value)
        
        # Ajusta os intervalos para variáveis inteiras
        for feature, bounds in rule_dict.items():
            if pd.api.types.is_integer_dtype(df_types[feature]):
                if bounds["min"] != -np.inf:
                    bounds["min"] = np.floor(bounds["min"] + 1)
                if bounds["max"] != np.inf:
                    bounds["max"] = np.floor(bounds["max"])

        formatted.append((rule_dict, predicted_class, gini_impurity))
    return formatted

def intervals_to_dataframe(formatted_intervals):
    """
    Converte a lista de intervalos formatados em um DataFrame do Pandas para fácil visualização.
    Inclui a impureza Gini e uma descrição detalhada da folha.
    """
    rows = []
    for i, (rules, predicted_class, gini_impurity) in enumerate(formatted_intervals):
        row = {"folha_id": i, "classe_predita": predicted_class, "gini_impureza": gini_impurity}
        description_parts = []
        for feature, bounds in rules.items():
            min_val = bounds["min"]
            max_val = bounds["max"]
            
            interval_str = ""
            if min_val == -np.inf and max_val == np.inf:
                interval_str = f"'{feature}' pode ser qualquer valor"
            elif min_val == -np.inf:
                interval_str = f"'{feature}' <= {max_val:.2f}"
            elif max_val == np.inf:
                interval_str = f"'{feature}' > {min_val:.2f}"
            else:
                interval_str = f"{min_val:.2f} < '{feature}' <= {max_val:.2f}"
            
            # Ajuste para features inteiras na descrição
            if pd.api.types.is_integer_dtype(pd.Series([min_val, max_val])):
                if min_val == -np.inf and max_val == np.inf:
                    interval_str = f"'{feature}' pode ser qualquer valor inteiro"
                elif min_val == -np.inf:
                    interval_str = f"'{feature}' <= {int(max_val)}"
                elif max_val == np.inf:
                    interval_str = f"'{feature}' > {int(min_val)}"
                else:
                    interval_str = f"{int(min_val)} < '{feature}' <= {int(max_val)}"

            description_parts.append(interval_str)
            
            # Adiciona a coluna de intervalo formatada para o DataFrame
            row[f"{feature}_intervalo"] = f"{min_val} a {max_val}"

        row["descricao_folha"] = " E ".join(description_parts) if description_parts else "Sem condições (folha raiz)"
        rows.append(row)
    
    return pd.DataFrame(rows).fillna("Livre")

def run_decision_tree_analysis(df, target_column, feature_columns, manual_intervals=None):
    """
    Orquestra a análise: treina a árvore, extrai, formata e exibe os intervalos.
    Permite a inserção de intervalos manuais.
    """
    # 1. Preparação dos dados
    X = df[feature_columns]
    y = df[target_column]
    
    # Garante que os nomes das classes sejam strings
    class_names = y.unique().astype(str)

    # 2. Treinamento do Modelo de Árvore de Decisão
    # Usamos max_depth para evitar uma árvore muito complexa e facilitar a visualização
    # Outros parâmetros de poda (min_samples_leaf, ccp_alpha) são importantes para a estabilidade
    tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_model.fit(X, y)

    # 3. Extração e Formatação dos Intervalos
    tree_ = tree_model.tree_
    leaf_intervals = get_leaf_intervals(tree_, feature_columns, class_names)
    formatted_intervals = format_intervals(leaf_intervals, X.dtypes)
    
    # 4. Criação do DataFrame final
    df_intervals = intervals_to_dataframe(formatted_intervals)

    # 5. Lógica para Intervalos Manuais
    if manual_intervals:
        print("\n--- Usando Intervalos Manuais ---")
        manual_rows = []
        for i, interval_set in enumerate(manual_intervals):
            row = {"folha_id": f"manual_{i}", "classe_predita": interval_set.get("classe_predita", "N/A")}
            description_parts = []
            for feature, bounds in interval_set.items():
                if feature != "classe_predita":
                    min_val = bounds[0]
                    max_val = bounds[1]
                    interval_str = ""
                    if min_val == -np.inf and max_val == np.inf:
                        interval_str = f"'{feature}' pode ser qualquer valor"
                    elif min_val == -np.inf:
                        interval_str = f"'{feature}' <= {max_val:.2f}"
                    elif max_val == np.inf:
                        interval_str = f"'{feature}' > {min_val:.2f}"
                    else:
                        interval_str = f"{min_val:.2f} < '{feature}' <= {max_val:.2f}"
                    
                    # Ajuste para features inteiras na descrição manual
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        if min_val == -np.inf and max_val == np.inf:
                            interval_str = f"'{feature}' pode ser qualquer valor inteiro"
                        elif min_val == -np.inf:
                            interval_str = f"'{feature}' <= {max_val}"
                        elif max_val == np.inf:
                            interval_str = f"'{feature}' > {min_val}"
                        else:
                            interval_str = f"{min_val} < '{feature}' <= {max_val}"

                    description_parts.append(interval_str)
                    row[f"{feature}_intervalo"] = f"{min_val} a {max_val}"
            row["descricao_folha"] = " E ".join(description_parts) if description_parts else "Sem condições (folha raiz)"
            manual_rows.append(row)
        df_intervals = pd.DataFrame(manual_rows).fillna("Livre")
    return df_intervals

def monitor_leaf_pd(df, df_intervals, target_column, feature_columns, date_column, leaf_id_column="folha_id", n_bootstrap=1000, alpha=0.05):
    """
    Simula o monitoramento da PD (Probabilidade de Default) por folha ao longo do tempo.
    Para cada folha, calcula a PD média, percentis via bootstrap e gera gráficos.
    """
    # Criar uma coluna para a folha_id no DataFrame original para simulação
    df["folha_atribuida"] = -1 # Valor padrão
    
    for idx, row in df_intervals.iterrows():
        folha_id = row[leaf_id_column]
        rules = {}
        for col in feature_columns:
            interval_str = row.get(f"{col}_intervalo")
            if interval_str and interval_str != "Livre":
                try:
                    min_val_str, max_val_str = interval_str.split(" a ")
                    min_val = float(min_val_str) if min_val_str != "-" else -np.inf
                    max_val = float(max_val_str) if max_val_str != "-" else np.inf
                    rules[col] = (min_val, max_val)
                except ValueError:
                    pass # Ignora intervalos manuais ou mal formatados

        # Aplica as regras para atribuir a folha
        if rules:
            condition = pd.Series([True] * len(df))
            for col, (min_val, max_val) in rules.items():
                if min_val != -np.inf:
                    condition = condition & (df[col] >= min_val)
                if max_val != np.inf:
                    condition = condition & (df[col] <= max_val)
            df.loc[condition, "folha_atribuida"] = folha_id

    # Calcular PD por folha, por mês e percentis via Bootstrap
    pd_results = []
    for mes in sorted(df[date_column].unique()):
        df_mes = df[df[date_column] == mes]
        for folha_id in df_mes["folha_atribuida"].unique():
            if folha_id == -1: # Clientes não atribuídos a nenhuma folha
                continue
            
            leaf_data = df_mes[df_mes["folha_atribuida"] == folha_id]
            if len(leaf_data) == 0: # Folha vazia no mês
                continue
                
            # Bootstrap para percentis detalhados
            bootstrap_samples = []
            for _ in range(n_bootstrap):
                sample = leaf_data.sample(n=len(leaf_data), replace=True)
                bootstrap_samples.append(sample[target_column].mean())
            
            # Calcular percentis específicos
            min_pd = np.min(bootstrap_samples)
            p1_pd = np.percentile(bootstrap_samples, 1)
            p5_pd = np.percentile(bootstrap_samples, 5)
            mean_pd = np.mean(bootstrap_samples)
            p95_pd = np.percentile(bootstrap_samples, 95)
            p99_pd = np.percentile(bootstrap_samples, 99)
            max_pd = np.max(bootstrap_samples)
            
            pd_results.append({
                "mes_referencia": mes,
                "folha_id": folha_id,
                "PD_min": min_pd,
                "PD_p1": p1_pd,
                "PD_p5": p5_pd,
                "PD_media": mean_pd,
                "PD_p95": p95_pd,
                "PD_p99": p99_pd,
                "PD_max": max_pd,
                "num_clientes": len(leaf_data)
            })
    
    pd_monitoramento_df = pd.DataFrame(pd_results)
    
    # Geração de gráficos por folha
    for folha_id in pd_monitoramento_df["folha_id"].unique():
        df_folha = pd_monitoramento_df[pd_monitoramento_df["folha_id"] == folha_id]
        plt.figure(figsize=(12, 6))
        sns.lineplot(x="mes_referencia", y="PD_media", data=df_folha, marker='o', label='PD Média', color='black')
        
        # Plotar linhas horizontais para os percentis e min/max
        plt.axhline(df_folha["PD_min"].mean(), color='gray', linestyle=':', label='PD Mínima (Média Geral)' )
        plt.axhline(df_folha["PD_p1"].mean(), color='purple', linestyle=':', label='PD P1 (Média Geral)' )
        plt.axhline(df_folha["PD_p5"].mean(), color='blue', linestyle=':', label='PD P5 (Média Geral)' )
        plt.axhline(df_folha["PD_p95"].mean(), color='orange', linestyle=':', label='PD P95 (Média Geral)' )
        plt.axhline(df_folha["PD_p99"].mean(), color='red', linestyle=':', label='PD P99 (Média Geral)' )
        plt.axhline(df_folha["PD_max"].mean(), color='darkred', linestyle=':', label='PD Máxima (Média Geral)' )

        # Linha para a média geral da folha (já existe, mas garantindo que seja visível)
        overall_mean_pd = df_folha["PD_media"].mean()
        plt.axhline(overall_mean_pd, color='green', linestyle='--', label=f'Média Geral da Folha: {overall_mean_pd:.2f}' )
        
        plt.title(f'Monitoramento de PD para Folha {folha_id}' )
        plt.xlabel('Mês de Referência')
        plt.ylabel('Probabilidade de Default (PD)' )
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'pd_monitoramento_folha_{folha_id}.png')
        plt.close()

    return pd_monitoramento_df

def test_feature_stability(df_current, df_past, feature_columns):
    """
    Testa a estabilidade das features entre dois períodos (atual e passado).
    Para features contínuas, usa o teste Kolmogorov-Smirnov (KS).
    Para features discretas/binárias, usa o teste Qui-Quadrado (Chi-Square).
    Retorna um DataFrame com os resultados dos testes.
    """
    
    stability_results = []
    
    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df_current[col]) and df_current[col].nunique() > 10: # Heurística para contínuas
            # Teste KS para features contínuas
            stat, p_value = kstest(df_current[col], df_past[col])
            stability_results.append({
                "feature": col,
                "tipo_teste": "Kolmogorov-Smirnov",
                "estatistica": stat,
                "p_valor": p_value,
                "estavel": "Sim" if p_value >= 0.05 else "Não" # p-valor > 0.05 indica que as distribuições são semelhantes
            })
        else: # Features discretas ou binárias
            # Teste Qui-Quadrado para features discretas/binárias
            # Garante que a tabela de contingência não seja trivial (pelo menos 2x2)
            # e que não haja valores zero que possam causar problemas no cálculo do qui-quadrado
            contingency_table = pd.crosstab(df_current[col], df_past[col])
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1 and (contingency_table > 0).all().all():
                chi2, p_value, _, _ = chi2_contingency(contingency_table)
                stability_results.append({
                    "feature": col,
                    "tipo_teste": "Qui-Quadrado",
                    "estatistica": chi2,
                    "p_valor": p_value,
                    "estavel": "Sim" if p_value >= 0.05 else "Não" # p-valor > 0.05 indica independência (distribuições semelhantes)
                })
            else:
                stability_results.append({
                    "feature": col,
                    "tipo_teste": "N/A",
                    "estatistica": np.nan,
                    "p_valor": np.nan,
                    "estavel": "Não Aplicável (tabela trivial ou com zeros)"
                })
                
    return pd.DataFrame(stability_results)

# --- Exemplo de Uso ---
if __name__ == "__main__":
    # Criação de um DataFrame de exemplo
    # Simulando dados de clientes para análise de PD (Probabilidade de Default)
    data = {
        'id_cliente': range(1, 201),
        'idade': np.random.randint(18, 70, 200),
        'salario_mensal': np.random.uniform(1000, 15000, 200).round(2),
        'parcelas_em_atraso': np.random.randint(0, 10, 200), # Feature inteira
        'limite_credito': np.random.uniform(500, 20000, 200),
        'mau_pagador': np.random.choice([0, 1], 200, p=[0.8, 0.2]), # 0: Bom, 1: Ruim
        'mes_referencia': pd.to_datetime(np.random.choice(pd.date_range('2024-01-01', '2024-06-01', freq='MS'), 200)) # Simula meses de referência
    }
    df_clientes = pd.DataFrame(data)

    # Definindo as colunas de features e a coluna alvo
    feature_columns = ['idade', 'salario_mensal', 'parcelas_em_atraso', 'limite_credito']
    target_column = 'mau_pagador'

    # --- Cenário 1: Análise Automática ---
    print("--- Análise com Intervalos Automáticos da Árvore ---")
    df_resultados_auto = run_decision_tree_analysis(df_clientes, target_column, feature_columns)
    print(df_resultados_auto)

    # --- Cenário 2: Análise com Intervalos Manuais ---
    # O usuário pode definir seus próprios intervalos com base no conhecimento do negócio.
    # Cada dicionário na lista representa uma "folha" ou segmento de cliente.
    # A chave 'classe_predita' é opcional e pode ser usada para nomear o segmento.
    # Os intervalos são definidos como tuplas (min_valor, max_valor).
    intervalos_manuais_exemplo = [
        {
            "classe_predita": "Bom Pagador Potencial",
            "parcelas_em_atraso_intervalo": (0, 1), # Exemplo de intervalo manual para feature inteira
            "salario_mensal_intervalo": (5000, np.inf) # Exemplo de intervalo manual para feature contínua
        },
        {
            "classe_predita": "Risco Elevado",
            "parcelas_em_atraso_intervalo": (3, np.inf), # Outro exemplo de intervalo manual
            "idade_intervalo": (18, 25)
        }
    ]
    
    # A função `run_decision_tree_analysis` atualmente apenas exibe os intervalos manuais.
    # Uma implementação completa poderia usar essas regras para segmentar o DataFrame original
    # e aplicar o monitoramento de PD a esses segmentos definidos manualmente.
    df_resultados_manual = run_decision_tree_analysis(df_clientes, target_column, feature_columns, manual_intervals=intervalos_manuais_exemplo)
    print("\n--- Análise com Intervalos Definidos Manualmente ---")
    print(df_resultados_manual)

    # --- Monitoramento de PD por folha (Simulado) ---
    # Esta seção demonstra como você pode monitorar a PD para cada folha.
    # O `df_clientes.copy()` é usado para evitar modificações no DataFrame original.
    print("\n--- Monitoramento de PD por Folha (Simulado) ---")
    # Adicionando uma coluna de mês para simular o monitoramento mensal
    df_clientes['mes_referencia_str'] = df_clientes['mes_referencia'].dt.strftime('%Y-%m')
    
    # Para o monitoramento, precisamos que o df_intervals tenha as folhas mapeadas para o df_clientes
    # Isso é feito dentro da função monitor_leaf_pd agora.
    pd_monitoramento = monitor_leaf_pd(df_clientes.copy(), df_resultados_auto, target_column, feature_columns, date_column='mes_referencia_str', leaf_id_column='folha_id')
    print(pd_monitoramento)

    # --- Teste de Estabilidade de Features (Simulado) ---
    print("\n--- Teste de Estabilidade de Features (Simulado) ---")
    # Criar um DataFrame de dados passados para simular a comparação
    data_passado = {
        'id_cliente': range(201, 401),
        'idade': np.random.randint(18, 75, 200), # Ligeira mudança na distribuição
        'salario_mensal': np.random.uniform(1200, 16000, 200).round(2),
        'parcelas_em_atraso': np.random.randint(0, 12, 200), # Ligeira mudança
        'limite_credito': np.random.uniform(600, 22000, 200),
        'mau_pagador': np.random.choice([0, 1], 200, p=[0.75, 0.25]), # Ligeira mudança na PD
        'mes_referencia': pd.to_datetime(np.random.choice(pd.date_range('2023-07-01', '2023-12-01', freq='MS'), 200))
    }
    df_clientes_passado = pd.DataFrame(data_passado)
    
    feature_stability_results = test_feature_stability(df_clientes[feature_columns], df_clientes_passado[feature_columns], feature_columns)
    print(feature_stability_results)

    # --- Considerações sobre Homogeneidade Temporal ---
    print("\n--- Considerações sobre Homogeneidade Temporal ---")
    print("Para garanti