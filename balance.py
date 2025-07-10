from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lit, rand, when
from pyspark.sql.types import FloatType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
import pandas as pd

# Inicialização da sessão Spark
spark = SparkSession.builder \
    .appName("ADASYN Implementation") \
    .getOrCreate()

def check_class_balance(df, target_col):
    """
    Verifica o balanceamento das classes no dataset.
    
    Retorna:
        Tuple (is_imbalanced, minority_class, minority_count, majority_count)
        - is_imbalanced: True se a razão minority/majority < 40%
        - minority_class: Classe com menos instâncias
        - minority_count: Contagem da classe minoritária
        - majority_count: Contagem da classe majoritária
    """
    class_counts = df.groupBy(target_col).count().collect()
    counts = {row[target_col]: row['count'] for row in class_counts}
    
    if len(counts) != 2:
        raise ValueError("A coluna alvo deve ter exatamente 2 classes (0 e 1)")
    
    minority_class = min(counts, key=counts.get)
    majority_class = max(counts, key=counts.get)
    imbalance_ratio = counts[minority_class] / counts[majority_class]
    
    print(f"Contagem de classes: {counts}")
    print(f"Razão de desbalanceamento: {imbalance_ratio:.2f}")
    
    return imbalance_ratio < 0.4, minority_class, counts[minority_class], counts[majority_class]

def calculate_k_neighbors(df, feature_col, k=5):
    """
    Calcula os k vizinhos mais próximos para cada ponto da classe minoritária.
    
    Parâmetros:
        df: DataFrame com os dados
        feature_col: Nome da coluna de features
        k: Número de vizinhos a considerar
        
    Retorna:
        distances: Matriz de distâncias para os vizinhos
        indices: Matriz de índices dos vizinhos
    """
    pandas_df = df.select(feature_col, "label").toPandas()
    minority_samples = pandas_df[pandas_df['label'] == 1][feature_col].apply(lambda x: x.toArray())
    
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(minority_samples.tolist())
    distances, indices = nbrs.kneighbors(minority_samples.tolist())
    
    return distances[:, 1:], indices[:, 1:]

def adasyn(df, target_col, feature_cols, k=5, beta=1.0):
    """
    Implementação do algoritmo ADASYN para balanceamento de classes.
    
    Parâmetros:
        df: DataFrame de entrada
        target_col: Nome da coluna target
        feature_cols: Lista de colunas de features
        k: Número de vizinhos próximos (explicação abaixo)
        beta: Grau de balanceamento (explicação abaixo)
        
    Retorna:
        DataFrame balanceado com amostras sintéticas
        
    =============================================================================
    Explicação dos Parâmetros k e β:
    
    k (Número de Vizinhos Próximos):
    - Controla quantos vizinhos da classe minoritária são considerados
    - Valores típicos: 5 ≤ k ≤ 15
    - k muito baixo (1-3): Pode causar overfitting
    - k muito alto (>20): Pode gerar amostras muito genéricas
    
    β (Grau de Balanceamento):
    - Controla quantas amostras sintéticas serão geradas
    - Fórmula: G = (N_majority - N_minority) × β
    - β = 0: Nenhuma amostra gerada
    - β = 1: Balanceamento perfeito (default)
    - β > 1: Classe minoritária pode se tornar maior
    =============================================================================
    """
    # Verificação de balanceamento
    is_imbalanced, minority_class, minority_count, majority_count = check_class_balance(df, target_col)
    
    if not is_imbalanced:
        print("Dataset já está balanceado. Retornando o dataset original.")
        return df
    
    print("Aplicando ADASYN para balanceamento...")
    
    # Preparação dos dados
    df = df.withColumnRenamed(target_col, "label")
    majority_df = df.filter(col("label") != minority_class)
    minority_df = df.filter(col("label") == minority_class)
    
    # Criação de vetores de features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_vector = assembler.transform(df).cache()
    
    # Cálculo dos vizinhos mais próximos
    distances, indices = calculate_k_neighbors(df_vector, "features", k)
    
    # Cálculo do número de amostras sintéticas
    G = (majority_count - minority_count) * beta
    
    # Cálculo da densidade de vizinhança
    r_i = np.sum(distances, axis=1) / k
    r_i_normalized = r_i / np.sum(r_i)
    g_i = np.round(r_i_normalized * G).astype(int)
    
    # Geração de amostras sintéticas
    synthetic_samples = []
    minority_pd = minority_df.select(feature_cols + ["label"]).toPandas()
    
    for i in range(len(minority_pd)):
        if g_i[i] > 0:
            for _ in range(g_i[i]):
                nn_idx = np.random.choice(indices[i])
                x_i = minority_pd.iloc[i][feature_cols].values
                x_zi = minority_pd.iloc[nn_idx][feature_cols].values
                delta = np.random.rand(len(feature_cols))
                x_new = x_i + delta * (x_zi - x_i)
                synthetic_samples.append(np.append(x_new, minority_class))
    
    # Criação do DataFrame com amostras sintéticas
    if synthetic_samples:
        synthetic_pd = pd.DataFrame(synthetic_samples, columns=feature_cols + ["label"])
        synthetic_df = spark.createDataFrame(synthetic_pd)
        
        for col_name in feature_cols:
            synthetic_df = synthetic_df.withColumn(col_name, col(col_name).cast(FloatType()))
        
        balanced_df = df.select(feature_cols + ["label"]).unionByName(synthetic_df)
    else:
        balanced_df = df
    
    # Verificação do novo balanceamento
    new_counts = balanced_df.groupBy("label").count().collect()
    new_counts = {row['label']: row['count'] for row in new_counts}
    print(f"Novas contagens de classes: {new_counts}")
    
    return balanced_df.withColumnRenamed("label", target_col)

# Exemplo de Uso
if __name__ == "__main__":
    # Criação de dataset de exemplo
    from pyspark.sql import Row
    
    data = [
        Row(feature1=1.0, feature2=2.0, feature3=3.0, target=0),
        Row(feature1=1.1, feature2=2.1, feature3=3.1, target=0),
        Row(feature1=1.2, feature2=2.2, feature3=3.2, target=0),
        Row(feature1=1.3, feature2=2.3, feature3=3.3, target=0),
        Row(feature1=1.4, feature2=2.4, feature3=3.4, target=0),
        Row(feature1=1.5, feature2=2.5, feature3=3.5, target=0),
        Row(feature1=1.6, feature2=2.6, feature3=3.6, target=0),
        Row(feature1=1.7, feature2=2.7, feature3=3.7, target=0),
        Row(feature1=1.8, feature2=2.8, feature3=3.8, target=0),
        Row(feature1=1.9, feature2=2.9, feature3=3.9, target=0),
        Row(feature1=4.0, feature2=5.0, feature3=6.0, target=1),
        Row(feature1=4.1, feature2=5.1, feature3=6.1, target=1)
    ]
    
    df = spark.createDataFrame(data)
    
    # Definição das colunas
    feature_cols = [c for c in df.columns if c != "target"]
    target_col = "target"
    
    # Aplicação do ADASYN
    print("\nAplicando ADASYN...")
    balanced_df = adasyn(df, target_col, feature_cols, k=5, beta=1.0)
    
    # Visualização dos resultados
    print("\nDataset balanceado:")
    balanced_df.show()
    
    # Preparação para modelagem com LightGBM
    from synapse.ml.lightgbm import LightGBMClassifier
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    lgbm_df = assembler.transform(balanced_df).cache()
    
    print("\nTreinando modelo LightGBM...")
    lgbm = LightGBMClassifier(
        objective="binary",
        featuresCol="features",
        labelCol=target_col,
        numLeaves=31,
        learningRate=0.1,
        nEstimators=100
    )
    
    model = lgbm.fit(lgbm_df)
    print("Modelo treinado com sucesso!")
    
    # Encerramento da sessão Spark
    spark.stop()
