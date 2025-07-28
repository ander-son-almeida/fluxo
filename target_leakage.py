"""
Script para detectar features que podem ser variáveis resposta (target leakage) em modelos de ML
Utiliza múltiplas técnicas para identificar features suspeitas que devem ser removidas
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm

class DetectorTargetLeakage:
    """
    Classe para detectar features que podem conter target leakage (variáveis resposta)
    
    Parâmetros de inicialização:
    :param threshold_importance: quantas vezes mais importante que a média para considerar suspeita (padrão: 3.0)
    :param threshold_corr: limite de correlação absoluta com o target (padrão: 0.95)
    :param threshold_auc: limite de AUC para classificação univariada (padrão: 0.95)
    :param threshold_entropia: limite de entropia normalizada (padrão: 0.1, menor = mais determinística)
    """
    
    def __init__(self, threshold_importance=3.0, threshold_corr=0.95, 
                 threshold_auc=0.95, threshold_entropia=0.1):
        self.thresholds = {
            'importancia': threshold_importance,
            'correlacao': threshold_corr,
            'auc': threshold_auc,
            'entropia': threshold_entropia
        }
        
    def detectar_leakage(self, X, y, verbose=True):
        """
        Método principal para detectar target leakage usando múltiplas técnicas
        
        Parâmetros:
        - X: DataFrame com as features
        - y: Series com o target
        - verbose: se True, imprime progresso e resultados
        
        Retorna:
        - lista de features suspeitas (flagadas por qualquer método)
        - dicionário com resultados detalhados por método
        - lista de features altamente suspeitas (flagadas por todos métodos)
        """
        resultados = {}
        features_suspeitas = set()
        
        # 1. Método da Importância Desproporcional
        if verbose: print("Aplicando método de Importância Desproporcional...")
        features_importancia = self._metodo_importancia(X, y)
        resultados['importancia'] = features_importancia
        features_suspeitas.update(features_importancia)
        
        # 2. Método da Correlação Alta
        if verbose: print("Aplicando método de Alta Correlação...")
        features_corr = self._metodo_correlacao(X, y)
        resultados['correlacao'] = features_corr
        features_suspeitas.update(features_corr)
        
        # 3. Método da AUC Univariada
        if verbose: print("Aplicando método de AUC Univariada...")
        features_auc = self._metodo_auc(X, y)
        resultados['auc'] = features_auc
        features_suspeitas.update(features_auc)
        
        # 4. Método da Entropia Condicional
        if verbose: print("Aplicando método de Entropia Condicional...")
        features_entropia = self._metodo_entropia(X, y)
        resultados['entropia'] = features_entropia
        features_suspeitas.update(features_entropia)
        
        # Features flagadas por todos os métodos (máxima confiança)
        features_confianca_total = (set(features_importancia) & 
                                   set(features_corr) & 
                                   set(features_auc) & 
                                   set(features_entropia))
        
        if verbose:
            print("\nResumo da Detecção:")
            print(f"- Método Importância flagou: {len(features_importancia)} features")
            print(f"- Método Correlação flagou: {len(features_corr)} features")
            print(f"- Método AUC flagou: {len(features_auc)} features")
            print(f"- Método Entropia flagou: {len(features_entropia)} features")
            print(f"\nTotal de features suspeitas únicas: {len(features_suspeitas)}")
            print(f"Features flagadas por todos métodos: {len(features_confianca_total)}")
        
        return list(features_suspeitas), resultados, list(features_confianca_total)
    
    def _metodo_importancia(self, X, y):
        """
        Identifica features com importância desproporcional no modelo
        
        1. Treina um modelo LightGBM com todas as features
        2. Calcula a importância de cada feature
        3. Compara com a importância média
        """
        modelo = LGBMClassifier(random_state=42)
        modelo.fit(X, y)
        
        # Obtém a importância de cada feature (ganho médio)
        importancias = modelo.feature_importances_
        
        # Calcula a importância média e define o threshold
        importancia_media = np.mean(importancias)
        threshold = self.thresholds['importancia'] * importancia_media
        
        # Retorna features com importância acima do threshold
        features_suspeitas = [coluna for coluna, imp in zip(X.columns, importancias) 
                             if imp > threshold]
        return features_suspeitas
    
    def _metodo_correlacao(self, X, y):
        """
        Identifica features com correlação extremamente alta com o target
        
        1. Calcula a correlação absoluta de Pearson entre cada feature e o target
        2. Filtra as features com correlação acima do threshold
        """
        # Calcula correlação absoluta para cada feature
        correlacoes = X.apply(lambda col: np.abs(np.corrcoef(col, y)[0, 1]))
        
        # Filtra features com correlação acima do threshold
        features_suspeitas = correlacoes[correlacoes > self.thresholds['correlacao']].index.tolist()
        return features_suspeitas
    
    def _metodo_auc(self, X, y):
        """
        Identifica features que podem prever o target muito bem sozinhas
        
        1. Para cada feature, treina um modelo usando apenas ela
        2. Calcula a AUC via validação cruzada
        3. Marca features com AUC muito alta
        """
        features_suspeitas = []
        
        # Itera sobre cada feature com barra de progresso
        for coluna in tqdm(X.columns, desc="Calculando AUC univariada"):
            # Usa apenas a feature atual
            dados_univariados = X[[coluna]]
            
            # Predições com validação cruzada para evitar overfitting
            predicoes = cross_val_predict(
                LGBMClassifier(random_state=42),
                dados_univariados,
                y,
                method='predict_proba',
                cv=5
            )[:, 1]  # Pega apenas as probabilidades da classe positiva
            
            # Calcula AUC
            auc = roc_auc_score(y, predicoes)
            
            # Verifica se AUC está acima do threshold
            if auc > self.thresholds['auc']:
                features_suspeitas.append(coluna)
                
        return features_suspeitas
    
    def _metodo_entropia(self, X, y):
        """
        Identifica features com baixa entropia condicional (quase determinísticas)
        
        1. Para cada valor da feature, calcula a distribuição do target
        2. Calcula a entropia condicional normalizada
        3. Marca features com entropia muito baixa
        """
        features_suspeitas = []
        
        for coluna in X.columns:
            # Cria DataFrame temporário para análise
            df_temp = pd.DataFrame({'feature': X[coluna], 'target': y})
            
            # Agrupa por valor da feature e calcula média do target
            agregado = df_temp.groupby('feature')['target'].mean()
            
            # Calcula a entropia condicional
            # Adiciona um pequeno epsilon (1e-10) para evitar log(0)
            entropia = -np.sum(
                agregado * np.log2(agregado + 1e-10) + 
                (1 - agregado) * np.log2(1 - agregado + 1e-10)
            )
            
            # Normaliza pela entropia máxima (1 para target binário)
            entropia_normalizada = entropia / 1
            
            # Verifica se a entropia está abaixo do threshold
            if entropia_normalizada < self.thresholds['entropia']:
                features_suspeitas.append(coluna)
                
        return features_suspeitas


# Exemplo de uso
if __name__ == "__main__":
    # Criando dados de exemplo (substitua pelos seus dados reais)
    from sklearn.datasets import make_classification
    
    print("\nCriando dados de exemplo com algumas features problemáticas...")
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=3, 
                             n_redundant=2, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y = pd.Series(y)
    
    # Adicionando algumas features com target leakage artificial
    X['vazamento_1'] = y.map(lambda x: x if np.random.random() > 0.05 else 1-x)  # 95% match
    X['vazamento_2'] = y * np.random.normal(1, 0.05, size=len(y))  # relação linear forte
    
    # Instanciando o detector
    detector = DetectorTargetLeakage()
    
    print("\nIniciando detecção de target leakage...")
    features_suspeitas, resultados_detalhados, features_confianca_total = detector.detectar_leakage(X, y)
    
    print("\nFeatures suspeitas (flagadas por qualquer método):")
    print(features_suspeitas)
    
    print("\nFeatures altamente suspeitas (flagadas por todos métodos):")
    print(features_confianca_total)
    
    print("\nDetalhes por método:")
    for metodo, features in resultados_detalhados.items():
        print(f"\n{metodo.upper()}:")
        print(features)