import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score
from scipy.stats import ks_2samp
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

class FeatureSelection:
    """
    Uma classe para realizar a seleção de features usando diversas técnicas metaheurísticas
    e o algoritmo Boruta, avaliando-as com uma métrica de desempenho e gerando um ranking.
    """
    def __init__(self, X, y, metric="ks", random_state=42):
        """
        Inicializa a classe FeatureSelection.

        Parâmetros:
        X (pd.DataFrame): O DataFrame contendo as features (variáveis independentes).
        y (pd.Series): A Série contendo a variável alvo (variável dependente).
        metric (str): A métrica de avaliação a ser utilizada para o modelo. Suportadas: 'ks', 'roc_auc', 'accuracy', 'f1', 'recall'.
                      Padrão: 'ks' (Kolmogorov-Smirnov).
        random_state (int): Semente para reprodutibilidade dos resultados.
        """
        self.X = X
        self.y = y
        self.metric = metric
        self.random_state = random_state
        self.results = [] # Lista para armazenar os resultados de cada técnica

    def _evaluate_features(self, selected_features):
        """
        Avalia um subconjunto de features usando um modelo de Regressão Logística e a métrica especificada.

        Parâmetros:
        selected_features (list): Uma lista de nomes de features selecionadas.

        Retorna:
        float: O valor médio da métrica de avaliação para o subconjunto de features.
        """
        if not selected_features:
            # Se nenhuma feature for selecionada, retorna -infinito para penalizar
            return -np.inf

        X_selected = self.X[selected_features]
        # StratifiedKFold para garantir que a proporção das classes seja mantida em cada fold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, test_idx in skf.split(X_selected, self.y):
            X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            # Modelo de Regressão Logística é um bom modelo base para avaliação de features
            model = LogisticRegression(random_state=self.random_state, solver="liblinear")
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            if self.metric == "ks":
                # Cálculo da estatística KS (Kolmogorov-Smirnov)
                group_0 = y_pred_proba[y_test == 0]
                group_1 = y_pred_proba[y_test == 1]
                if len(group_0) > 0 and len(group_1) > 0:
                    ks_statistic = ks_2samp(group_0, group_1).statistic
                    scores.append(ks_statistic)
                else:
                    # Se um grupo estiver vazio, o KS não pode ser calculado, penaliza
                    scores.append(-np.inf)
            elif self.metric == "roc_auc":
                scores.append(roc_auc_score(y_test, y_pred_proba))
            elif self.metric == "accuracy":
                y_pred = model.predict(X_test)
                scores.append(accuracy_score(y_test, y_pred))
            elif self.metric == "f1":
                y_pred = model.predict(X_test)
                scores.append(f1_score(y_test, y_pred))
            elif self.metric == "recall":
                y_pred = model.predict(X_test)
                scores.append(recall_score(y_test, y_pred))
            else:
                raise ValueError(f"Métrica de avaliação não suportada: {self.metric}")

        return np.mean(scores)

    def get_ranking(self, target_metric, target_features):
        """
        Calcula o score composto para cada técnica e gera um ranking.

        Parâmetros:
        target_metric (float): O valor da métrica obtido em um modelo de referência.
        target_features (int): O número de features utilizadas no modelo de referência.

        Retorna:
        pd.DataFrame: Um DataFrame com as técnicas, seus valores de métrica, número de features,
                      features selecionadas e o score composto, ordenado do melhor para o pior score.
        """
        df_results = pd.DataFrame(self.results)
        
        # Ajusta o número de features e o valor da métrica para evitar divisão por zero ou valores inválidos
        df_results["Número de Features Ajustado"] = df_results["Número de Features"].apply(lambda x: max(x, 1))
        df_results["Valor da Métrica Ajustado"] = df_results["Valor da Métrica"].apply(lambda x: x if x != -np.inf else 0)

        # Cálculo do Score Composto:
        # Score = 0.7 * (valor_da_metrica / valor_metrica_target) + 0.3 * (num_features_target / num_features_utilizadas)
        # Favorece métricas iguais ou superiores ao modelo de referência usando menos features.
        df_results["Score Composto"] = df_results.apply(
            lambda row: 0.7 * (row["Valor da Métrica Ajustado"] / target_metric) + \
                        0.3 * (target_features / row["Número de Features Ajustado"]),
            axis=1
        )
        return df_results.sort_values(by="Score Composto", ascending=False)

    def run_boruta(self):
        """
        Executa o algoritmo Boruta para seleção de features.
        Boruta é um algoritmo de seleção de features baseado em Random Forest, que busca
        identificar todas as features relevantes, em vez de apenas um subconjunto ótimo.
        """
        print("Executando Boruta...")
        # BorutaPy espera arrays NumPy
        X_np = self.X.values
        y_np = self.y.values

        # Inicializa o Random Forest Classifier para ser usado pelo Boruta
        # n_jobs=-1: usa todos os núcleos da CPU para paralelizar
        # class_weight="balanced": ajusta pesos inversamente proporcionais à frequência das classes
        # max_depth=5: profundidade máxima da árvore para evitar overfitting e acelerar
        rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5, random_state=self.random_state)
        
        # Inicializa o BorutaPy
        # n_estimators="auto": determina o número de árvores automaticamente
        boruta_selector = BorutaPy(rf, n_estimators="auto", random_state=self.random_state)

        # Treina o Boruta
        boruta_selector.fit(X_np, y_np)

        # Obtém a máscara das features selecionadas
        selected_features_mask = boruta_selector.support_
        selected_features = list(self.X.columns[selected_features_mask])

        metric_value = self._evaluate_features(selected_features)
        num_features = len(selected_features)
        self.results.append({
            "Técnica": "Boruta",
            "Valor da Métrica": metric_value,
            "Número de Features": num_features,
            "Features Selecionadas": selected_features
        })
        print(f"Boruta concluído. Métrica: {metric_value:.4f}, Features: {num_features}")

    # --- Implementações das Metaheurísticas ---

    def run_binary_bat_algorithm(self, n_iterations=50, pop_size=10):
        """
        Executa o Algoritmo Binário do Morcego (Binary Bat Algorithm - BBA) para seleção de features.
        O BBA é inspirado no comportamento de ecolocalização dos morcegos.

        Parâmetros:
        n_iterations (int): Número de iterações (gerações) do algoritmo.
        pop_size (int): Tamanho da população de morcegos (soluções).
        """
        print("Executando Binary Bat Algorithm...")
        num_features = self.X.shape[1]
        
        # Inicializa a melhor solução encontrada até agora aleatoriamente
        best_features = np.random.randint(0, 2, size=num_features)
        best_metric = self._evaluate_features(list(self.X.columns[best_features.astype(bool)]))

        # Inicializa a população de morcegos (vetores binários representando subconjuntos de features)
        population = np.random.randint(0, 2, size=(pop_size, num_features))
        velocities = np.zeros((pop_size, num_features)) # Velocidades dos morcegos
        frequencies = np.zeros(pop_size) # Frequências de pulso dos morcegos
        loudness = 1.0 # A: Intensidade do som (loudness)
        pulse_rate = 0.9 # r: Taxa de emissão de pulso

        for iteration in range(n_iterations):
            for i in range(pop_size):
                # Atualiza frequência, velocidade e posição
                # A frequência é simplificada aqui, em uma implementação completa, varia entre f_min e f_max
                frequencies[i] = 0.5 * np.random.rand()
                # A velocidade é atualizada com base na diferença entre a posição atual e a melhor global
                velocities[i] = velocities[i] + (population[i] - best_features) * frequencies[i]
                
                # Função Sigmoid para mapear a velocidade para uma probabilidade binária
                # Isso transforma o movimento contínuo em uma decisão binária (0 ou 1)
                s_function = 1 / (1 + np.exp(-velocities[i]))
                # Nova posição é determinada probabilisticamente
                new_position = (s_function > np.random.rand(num_features)).astype(int)

                # Busca local: se um morcego emite um pulso (probabilidade > pulse_rate),
                # ele realiza uma busca local (aqui, um simples flip de bit)
                if np.random.rand() > pulse_rate:
                    # Flip de bit aleatório em 10% das features para simular busca local
                    new_position = np.abs(new_position - (np.random.rand(num_features) < 0.1).astype(int))

                current_features = list(self.X.columns[new_position.astype(bool)])
                current_metric = self._evaluate_features(current_features)

                # Aceita a nova solução se for melhor e se a intensidade do som permitir
                if current_metric > best_metric and np.random.rand() < loudness:
                    best_metric = current_metric
                    best_features = new_position
                    # A intensidade do som e a taxa de pulso são atualizadas para simular o comportamento do morcego
                    loudness *= 0.9 # Diminui a intensidade do som
                    pulse_rate *= (1 - np.exp(-0.1 * iteration)) # Aumenta a taxa de pulso

            if iteration % 10 == 0:
                print(f"  Iteração {iteration}/{n_iterations}, Melhor Métrica: {best_metric:.4f}")

        selected_features = list(self.X.columns[best_features.astype(bool)])
        num_features = len(selected_features)
        self.results.append({
            "Técnica": "Binary Bat Algorithm",
            "Valor da Métrica": best_metric,
            "Número de Features": num_features,
            "Features Selecionadas": selected_features
        })
        print(f"Binary Bat Algorithm concluído. Métrica: {best_metric:.4f}, Features: {num_features}")

    def run_cuckoo_search(self, n_iterations=50, pop_size=10):
        """
        Executa o Algoritmo de Busca do Cuco (Cuckoo Search - CS) para seleção de features.
        O CS é inspirado no parasitismo de cria de algumas espécies de cucos.

        Parâmetros:
        n_iterations (int): Número de iterações (gerações) do algoritmo.
        pop_size (int): Tamanho da população de ninhos (soluções).
        """
        print("Executando Cuckoo Search...")
        num_features = self.X.shape[1]
        
        # Inicializa a melhor solução encontrada até agora aleatoriamente
        best_features = np.random.randint(0, 2, size=num_features)
        best_metric = self._evaluate_features(list(self.X.columns[best_features.astype(bool)]))

        # Inicializa os ninhos (vetores binários representando subconjuntos de features)
        nests = np.random.randint(0, 2, size=(pop_size, num_features))
        pa = 0.25 # pa: Probabilidade de um ninho ser descoberto por um hospedeiro (ovo alienígena)

        for iteration in range(n_iterations):
            for i in range(pop_size):
                # Gera uma nova solução (ovo) usando um voo de Lévy
                # O voo de Lévy é um tipo de caminhada aleatória que permite saltos maiores,
                # favorecendo a exploração do espaço de busca.
                levy = np.random.normal(0, 1, num_features) # Simplificação do voo de Lévy
                new_nest = nests[i] + levy.astype(int) # Adiciona o voo de Lévy à posição atual
                new_nest = np.clip(new_nest, 0, 1) # Garante que as features permaneçam binárias (0 ou 1)

                current_features = list(self.X.columns[new_nest.astype(bool)])
                current_metric = self._evaluate_features(current_features)

                # Avalia o novo ninho: se a nova solução for melhor que a atual do ninho, substitui
                if current_metric > self._evaluate_features(list(self.X.columns[nests[i].astype(bool)])):
                    nests[i] = new_nest

                # Atualiza a melhor solução global encontrada até agora
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_features = new_nest

            # Abandona alguns ninhos e constrói novos com base na probabilidade 'pa'
            # Isso simula a descoberta de ovos alienígenas pelos pássaros hospedeiros
            if np.random.rand() < pa:
                nests[np.random.randint(pop_size)] = np.random.randint(0, 2, size=num_features)

            if iteration % 10 == 0:
                print(f"  Iteração {iteration}/{n_iterations}, Melhor Métrica: {best_metric:.4f}")

        selected_features = list(self.X.columns[best_features.astype(bool)])
        num_features = len(selected_features)
        self.results.append({
            "Técnica": "Cuckoo Search",
            "Valor da Métrica": best_metric,
            "Número de Features": num_features,
            "Features Selecionadas": selected_features
        })
        print(f"Cuckoo Search concluído. Métrica: {best_metric:.4f}, Features: {num_features}")

    def run_equilibrium_optimizer(self, n_iterations=50, pop_size=10):
        """
        Executa o Otimizador de Equilíbrio (Equilibrium Optimizer - EO) para seleção de features.
        O EO é inspirado em modelos de balanço de massa para estimar concentrações de partículas.

        Parâmetros:
        n_iterations (int): Número de iterações (gerações) do algoritmo.
        pop_size (int): Tamanho da população de partículas (soluções).
        """
        print("Executando Equilibrium Optimizer...")
        num_features = self.X.shape[1]
        
        # Inicializa a melhor solução encontrada até agora aleatoriamente
        best_features = np.random.randint(0, 2, size=num_features)
        best_metric = self._evaluate_features(list(self.X.columns[best_features.astype(bool)]))

        # Inicializa a população de partículas (vetores binários)
        population = np.random.randint(0, 2, size=(pop_size, num_features))

        for iteration in range(n_iterations):
            # Atualiza o pool de equilíbrio (simplificado para as 4 melhores soluções)
            # O pool de equilíbrio representa os estados de equilíbrio ideais para as concentrações.
            eq_pool = population[np.argsort([self._evaluate_features(list(self.X.columns[p.astype(bool)])) for p in population])[-4:]]
            C_eq = np.mean(eq_pool, axis=0) # Concentração de equilíbrio média

            for i in range(pop_size):
                # Gera uma nova solução baseada na concentração de equilíbrio
                F = np.random.rand() # Fator de geração
                r1 = np.random.rand()
                r2 = np.random.rand()
                
                # Equação de atualização simplificada para o espaço binário
                # A nova posição é influenciada pela concentração de equilíbrio e por fatores aleatórios
                new_position = population[i] + F * (C_eq - population[i]) * r1 + (np.random.rand(num_features) < r2).astype(int)
                new_position = np.clip(new_position, 0, 1) # Garante que as features permaneçam binárias

                current_features = list(self.X.columns[new_position.astype(bool)])
                current_metric = self._evaluate_features(current_features)

                # Se a nova solução for melhor, atualiza a posição da partícula
                if current_metric > self._evaluate_features(list(self.X.columns[population[i].astype(bool)])):
                    population[i] = new_position

                # Atualiza a melhor solução global
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_features = new_position

            if iteration % 10 == 0:
                print(f"  Iteração {iteration}/{n_iterations}, Melhor Métrica: {best_metric:.4f}")

        selected_features = list(self.X.columns[best_features.astype(bool)])
        num_features = len(selected_features)
        self.results.append({
            "Técnica": "Equilibrium Optimizer",
            "Valor da Métrica": best_metric,
            "Número de Features": num_features,
            "Features Selecionadas": selected_features
        })
        print(f"Equilibrium Optimizer concluído. Métrica: {best_metric:.4f}, Features: {num_features}")

    def run_genetic_algorithm(self, n_iterations=50, pop_size=10):
        """
        Executa o Algoritmo Genético (Genetic Algorithm - GA) para seleção de features.
        O GA é inspirado nos princípios da seleção natural e da genética.

        Parâmetros:
        n_iterations (int): Número de iterações (gerações) do algoritmo.
        pop_size (int): Tamanho da população de indivíduos (soluções).
        """
        print("Executando Genetic Algorithm...")
        num_features = self.X.shape[1]
        
        # Inicializa a melhor solução encontrada até agora aleatoriamente
        best_features = np.random.randint(0, 2, size=num_features)
        best_metric = self._evaluate_features(list(self.X.columns[best_features.astype(bool)]))

        # Inicializa a população de indivíduos (vetores binários)
        population = np.random.randint(0, 2, size=(pop_size, num_features))

        for iteration in range(n_iterations):
            # Avalia a aptidão (fitness) de cada indivíduo na população
            fitness = [self._evaluate_features(list(self.X.columns[p.astype(bool)])) for p in population]

            # Seleção: Escolhe os indivíduos mais aptos para reprodução.
            # Aqui, usa-se a Seleção por Roleta (Roulette Wheel Selection).
            total_fitness = sum(fitness)
            if total_fitness == 0:
                # Evita divisão por zero se todos os fitness forem 0 (caso raro)
                probabilities = np.ones(pop_size) / pop_size
            else:
                probabilities = np.array(fitness) / total_fitness
            
            # Seleciona indivíduos com base nas probabilidades de aptidão
            selected_indices = np.random.choice(pop_size, size=pop_size, p=probabilities)
            selected_population = population[selected_indices]

            new_population = []
            for i in range(0, pop_size, 2):
                parent1 = selected_population[i]
                # Garante que haja um segundo pai, mesmo se pop_size for ímpar
                parent2 = selected_population[i+1] if i+1 < pop_size else selected_population[0]

                # Crossover (Recombinação): Combina material genético dos pais para criar filhos.
                # Crossover de Ponto Único (Single Point Crossover): Divide os pais em um ponto e troca as partes.
                crossover_point = np.random.randint(1, num_features) # Ponto de corte aleatório
                child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

                # Mutação: Introduz pequenas alterações aleatórias no material genético.
                # Mutação por Flip de Bit (Bit Flip Mutation): Inverte um bit com uma pequena probabilidade.
                mutation_rate = 0.01 # Taxa de mutação (1% de chance de um bit ser invertido)
                child1 = np.abs(child1 - (np.random.rand(num_features) < mutation_rate).astype(int))
                child2 = np.abs(child2 - (np.random.rand(num_features) < mutation_rate).astype(int))

                new_population.extend([child1, child2])
            
            # Atualiza a população com os novos indivíduos gerados
            population = np.array(new_population[:pop_size]) # Garante que o tamanho da população seja mantido

            # Atualiza a melhor solução global encontrada
            current_best_idx = np.argmax(fitness)
            current_best_metric = fitness[current_best_idx]
            if current_best_metric > best_metric:
                best_metric = current_best_metric
                best_features = population[current_best_idx]

            if iteration % 10 == 0:
                print(f"  Iteração {iteration}/{n_iterations}, Melhor Métrica: {best_metric:.4f}")

        selected_features = list(self.X.columns[best_features.astype(bool)])
        num_features = len(selected_features)
        self.results.append({
            "Técnica": "Genetic Algorithm",
            "Valor da Métrica": best_metric,
            "Número de Features": num_features,
            "Features Selecionadas": selected_features
        })
        print(f"Genetic Algorithm concluído. Métrica: {best_metric:.4f}, Features: {num_features}")

    def run_gravitational_search_algorithm(self, n_iterations=50, pop_size=10):
        """
        Executa o Algoritmo de Busca Gravitacional (Gravitational Search Algorithm - GSA) para seleção de features.
        O GSA é inspirado na lei da gravitação universal de Newton, onde os agentes (soluções) se atraem
        proporcionalmente à sua massa (qualidade da solução).

        Parâmetros:
        n_iterations (int): Número de iterações (gerações) do algoritmo.
        pop_size (int): Tamanho da população de agentes (soluções).
        """
        print("Executando Gravitational Search Algorithm...")
        num_features = self.X.shape[1]
        
        # Inicializa a melhor solução encontrada até agora aleatoriamente
        best_features = np.random.randint(0, 2, size=num_features)
        best_metric = self._evaluate_features(list(self.X.columns[best_features.astype(bool)]))

        # Inicializa os agentes (vetores binários)
        agents = np.random.randint(0, 2, size=(pop_size, num_features))
        velocities = np.zeros((pop_size, num_features)) # Velocidades dos agentes
        masses = np.ones(pop_size) # Massas dos agentes (inicialmente uniformes)

        G0 = 100 # Constante gravitacional inicial

        for iteration in range(n_iterations):
            # Calcula a aptidão (fitness) e atualiza as massas
            fitness = np.array([self._evaluate_features(list(self.X.columns[a.astype(bool)])) for a in agents])
            best_idx = np.argmax(fitness) # Índice do melhor agente
            worst_idx = np.argmin(fitness) # Índice do pior agente

            if fitness[best_idx] > best_metric:
                best_metric = fitness[best_idx]
                best_features = agents[best_idx]

            # Normaliza o fitness para calcular a massa
            # Agentes com maior fitness (melhores soluções) têm maior massa e, portanto, maior atração.
            min_fitness = np.min(fitness)
            max_fitness = np.max(fitness)
            if max_fitness == min_fitness:
                masses = np.ones(pop_size) # Evita divisão por zero se todos os fitness forem iguais
            else:
                masses = (fitness - min_fitness) / (max_fitness - min_fitness)
            masses = masses / np.sum(masses) # Normaliza as massas para que a soma seja 1

            # Atualiza a constante gravitacional G (diminui com o tempo para favorecer a exploração no início e a explotação no final)
            G = G0 * np.exp(-iteration / n_iterations)

            # Calcula as forças e atualiza velocidades e posições
            for i in range(pop_size):
                total_force = np.zeros(num_features)
                for j in range(pop_size):
                    if i != j:
                        # Calcula a distância entre os agentes
                        R = np.linalg.norm(agents[i] - agents[j]) + 1e-9 # Adiciona um pequeno valor para evitar divisão por zero
                        # Calcula a força gravitacional
                        force = G * masses[i] * masses[j] / R * (agents[j] - agents[i])
                        total_force += np.random.rand() * force # Adiciona aleatoriedade à força
                
                # Calcula a aceleração e atualiza a velocidade
                acceleration = total_force / (masses[i] + 1e-9) # Adiciona um pequeno valor para evitar divisão por zero
                velocities[i] = np.random.rand() * velocities[i] + acceleration # Atualiza velocidade com inércia e aceleração

                # Aplica a função Sigmoid para converter a velocidade contínua em uma decisão binária
                s_function = 1 / (1 + np.exp(-velocities[i]))
                agents[i] = (s_function > np.random.rand(num_features)).astype(int) # Atualiza a posição do agente

            if iteration % 10 == 0:
                print(f"  Iteração {iteration}/{n_iterations}, Melhor Métrica: {best_metric:.4f}")

        selected_features = list(self.X.columns[best_features.astype(bool)])
        num_features = len(selected_features)
        self.results.append({
            "Técnica": "Gravitational Search Algorithm",
            "Valor da Métrica": best_metric,
            "Número de Features": num_features,
            "Features Selecionadas": selected_features
        })
        print(f"Gravitational Search Algorithm concluído. Métrica: {best_metric:.4f}, Features: {num_features}")

    def run_grey_wolf_optimizer(self, n_iterations=50, pop_size=10):
        """
        Executa o Otimizador de Lobo Cinzento (Grey Wolf Optimizer - GWO) para seleção de features.
        O GWO é inspirado na hierarquia social e no comportamento de caça dos lobos cinzentos.

        Parâmetros:
        n_iterations (int): Número de iterações (gerações) do algoritmo.
        pop_size (int): Tamanho da população de lobos (soluções).
        """
        print("Executando Grey Wolf Optimizer...")
        num_features = self.X.shape[1]
        
        # Inicializa a melhor solução encontrada até agora aleatoriamente
        best_features = np.random.randint(0, 2, size=num_features)
        best_metric = self._evaluate_features(list(self.X.columns[best_features.astype(bool)]))

        # Inicializa a população de lobos (vetores binários)
        wolves = np.random.randint(0, 2, size=(pop_size, num_features))

        for iteration in range(n_iterations):
            # Calcula a aptidão (fitness) e identifica os lobos alfa, beta e delta
            # Alfa (melhor), Beta (segundo melhor), Delta (terceiro melhor)
            fitness = np.array([self._evaluate_features(list(self.X.columns[w.astype(bool)])) for w in wolves])
            sorted_indices = np.argsort(fitness)[::-1] # Índices em ordem decrescente de fitness

            alpha_wolf = wolves[sorted_indices[0]]
            beta_wolf = wolves[sorted_indices[1]]
            delta_wolf = wolves[sorted_indices[2]]

            if fitness[sorted_indices[0]] > best_metric:
                best_metric = fitness[sorted_indices[0]]
                best_features = alpha_wolf

            # Atualiza os parâmetros 'a', 'C' e 'D'
            # 'a' diminui linearmente de 2 para 0, controlando a exploração vs. explotação
            a = 2 - iteration * (2 / n_iterations)

            for i in range(pop_size):
                # A1, A2, A3: Componentes de ataque (aleatórios)
                # C1, C2, C3: Componentes de exploração (aleatórios)
                A1, A2, A3 = 2 * a * np.random.rand() - a, 2 * a * np.random.rand() - a, 2 * a * np.random.rand() - a
                C1, C2, C3 = 2 * np.random.rand(), 2 * np.random.rand(), 2 * np.random.rand()

                # Calcula a distância entre o lobo atual e os lobos alfa, beta, delta
                D_alpha = np.abs(C1 * alpha_wolf - wolves[i])
                X1 = alpha_wolf - A1 * D_alpha # Posição sugerida pelo lobo alfa

                D_beta = np.abs(C2 * beta_wolf - wolves[i])
                X2 = beta_wolf - A2 * D_beta # Posição sugerida pelo lobo beta

                D_delta = np.abs(C3 * delta_wolf - wolves[i])
                X3 = delta_wolf - A3 * D_delta # Posição sugerida pelo lobo delta

                # Atualiza a posição do lobo (simplificado para o espaço binário)
                # A nova posição é a média das posições sugeridas pelos três melhores lobos
                new_position = (X1 + X2 + X3) / 3
                # Aplica a função Sigmoid para converter a posição contínua em uma decisão binária
                new_position = (1 / (1 + np.exp(-new_position)) > np.random.rand(num_features)).astype(int)

                current_features = list(self.X.columns[new_position.astype(bool)])
                current_metric = self._evaluate_features(current_features)

                # Se a nova posição for melhor, atualiza a posição do lobo
                if current_metric > self._evaluate_features(list(self.X.columns[wolves[i].astype(bool)])):
                    wolves[i] = new_position

            if iteration % 10 == 0:
                print(f"  Iteração {iteration}/{n_iterations}, Melhor Métrica: {best_metric:.4f}")

        selected_features = list(self.X.columns[best_features.astype(bool)])
        num_features = len(selected_features)
        self.results.append({
            "Técnica": "Grey Wolf Optimizer",
            "Valor da Métrica": best_metric,
            "Número de Features": num_features,
            "Features Selecionadas": selected_features
        })
        print(f"Grey Wolf Optimizer concluído. Métrica: {best_metric:.4f}, Features: {num_features}")

    def run_harmony_search(self, n_iterations=50, pop_size=10):
        """
        Executa o Algoritmo de Busca Harmônica (Harmony Search - HS) para seleção de features.
        O HS é inspirado no processo de improvisação musical, onde músicos buscam a harmonia perfeita.

        Parâmetros:
        n_iterations (int): Número de iterações (gerações) do algoritmo.
        pop_size (int): Tamanho da memória harmônica (número de soluções).
        """
        print("Executando Harmony Search...")
        num_features = self.X.shape[1]
        
        # Inicializa a melhor solução encontrada até agora aleatoriamente
        best_features = np.random.randint(0, 2, size=num_features)
        best_metric = self._evaluate_features(list(self.X.columns[best_features.astype(bool)]))

        # Inicializa a memória harmônica (vetores binários)
        harmony_memory = np.random.randint(0, 2, size=(pop_size, num_features))
        hmcr = 0.9 # HMCR (Harmony Memory Considering Rate): Probabilidade de usar um valor da memória harmônica
        par = 0.1 # PAR (Pitch Adjustment Rate): Probabilidade de ajustar o tom (mutação)

        for iteration in range(n_iterations):
            # Gera uma nova harmonia (solução)
            new_harmony = np.zeros(num_features)
            for j in range(num_features):
                if np.random.rand() < hmcr:
                    # Escolhe um valor da memória harmônica
                    new_harmony[j] = harmony_memory[np.random.randint(pop_size), j]
                    if np.random.rand() < par:
                        # Ajuste de tom (mutação): inverte o bit com uma pequena probabilidade
                        new_harmony[j] = 1 - new_harmony[j]
                else:
                    # Gera um valor aleatório (exploração)
                    new_harmony[j] = np.random.randint(0, 2)
            
            current_features = list(self.X.columns[new_harmony.astype(bool)])
            current_metric = self._evaluate_features(current_features)

            # Atualiza a memória harmônica: se a nova harmonia for melhor que a pior na memória,
            # a pior é substituída pela nova.
            worst_harmony_idx = np.argmin([self._evaluate_features(list(self.X.columns[h.astype(bool)])) for h in harmony_memory])
            if current_metric > self._evaluate_features(list(self.X.columns[harmony_memory[worst_harmony_idx].astype(bool)])):
                harmony_memory[worst_harmony_idx] = new_harmony

            # Atualiza a melhor solução global
            if current_metric > best_metric:
                best_metric = current_metric
                best_features = new_harmony

            if iteration % 10 == 0:
                print(f"  Iteração {iteration}/{n_iterations}, Melhor Métrica: {best_metric:.4f}")

        selected_features = list(self.X.columns[best_features.astype(bool)])
        num_features = len(selected_features)
        self.results.append({
            "Técnica": "Harmony Search",
            "Valor da Métrica": best_metric,
            "Número de Features": num_features,
            "Features Selecionadas": selected_features
        })
        print(f"Harmony Search concluído. Métrica: {best_metric:.4f}, Features: {num_features}")

    def run_memetic_algorithm(self, n_iterations=50, pop_size=10):
        """
        Executa o Algoritmo Memético (Memetic Algorithm - MA) para seleção de features.
        O MA combina algoritmos genéticos (busca global) com busca local (otimização local).

        Parâmetros:
        n_iterations (int): Número de iterações (gerações) do algoritmo.
        pop_size (int): Tamanho da população de indivíduos (soluções).
        """
        print("Executando Memetic Algorithm...")
        num_features = self.X.shape[1]
        
        # Inicializa a melhor solução encontrada até agora aleatoriamente
        best_features = np.random.randint(0, 2, size=num_features)
        best_metric = self._evaluate_features(list(self.X.columns[best_features.astype(bool)]))

        # Inicializa a população de indivíduos (vetores binários)
        population = np.random.randint(0, 2, size=(pop_size, num_features))

        for iteration in range(n_iterations):
            # Avalia a aptidão (fitness) de cada indivíduo
            fitness = [self._evaluate_features(list(self.X.columns[p.astype(bool)])) for p in population]

            # Seleção: Seleção por Torneio (Tournament Selection)
            # Escolhe um subconjunto aleatório de indivíduos (torneio) e seleciona o melhor deles.
            new_population_selected = []
            for _ in range(pop_size):
                tournament_size = 3 # Tamanho do torneio
                competitors_indices = np.random.choice(pop_size, size=tournament_size, replace=False)
                winner_index = competitors_indices[np.argmax(np.array(fitness)[competitors_indices])]
                new_population_selected.append(population[winner_index])
            population = np.array(new_population_selected)

            # Crossover (Recombinação): Crossover Uniforme (Uniform Crossover)
            # Cada bit é trocado entre os pais com uma probabilidade definida.
            crossover_rate = 0.8 # Taxa de crossover
            offspring_population = []
            for i in range(0, pop_size, 2):
                parent1 = population[i]
                parent2 = population[i+1] if i+1 < pop_size else population[0]

                if np.random.rand() < crossover_rate:
                    mask = np.random.randint(0, 2, size=num_features) # Máscara binária para troca de bits
                    child1 = mask * parent1 + (1 - mask) * parent2
                    child2 = mask * parent2 + (1 - mask) * parent1
                else:
                    child1 = parent1
                    child2 = parent2

                offspring_population.append(child1)
                if i+1 < pop_size:
                    offspring_population.append(child2)
            
            population = np.array(offspring_population[:pop_size])

            # Mutação: Mutação por Flip de Bit (Bit Flip Mutation)
            mutation_rate = 0.01 # Taxa de mutação
            for i in range(pop_size):
                population[i] = np.abs(population[i] - (np.random.rand(num_features) < mutation_rate).astype(int))

            # Busca Local (Otimização Local): Hill Climbing simples em um subconjunto da população
            # Melhora as soluções individualmente após o crossover e mutação.
            for i in range(pop_size // 2): # Aplica busca local a metade da população
                current_solution = population[i].copy()
                current_metric = self._evaluate_features(list(self.X.columns[current_solution.astype(bool)]))
                
                # Tenta inverter um bit de cada vez e aceita se houver melhora
                for j in range(num_features):
                    neighbor = current_solution.copy()
                    neighbor[j] = 1 - neighbor[j] # Inverte o bit
                    neighbor_metric = self._evaluate_features(list(self.X.columns[neighbor.astype(bool)]))
                    if neighbor_metric > current_metric:
                        current_solution = neighbor
                        current_metric = neighbor_metric
                population[i] = current_solution

            # Atualiza a melhor solução global
            current_best_idx = np.argmax(fitness)
            current_best_metric = fitness[current_best_idx]
            if current_best_metric > best_metric:
                best_metric = current_best_metric
                best_features = population[current_best_idx]

            if iteration % 10 == 0:
                print(f"  Iteração {iteration}/{n_iterations}, Melhor Métrica: {best_metric:.4f}")

        selected_features = list(self.X.columns[best_features.astype(bool)])
        num_features = len(selected_features)
        self.results.append({
            "Técnica": "Memetic Algorithm",
            "Valor da Métrica": best_metric,
            "Número de Features": num_features,
            "Features Selecionadas": selected_features
        })
        print(f"Memetic Algorithm concluído. Métrica: {best_metric:.4f}, Features: {num_features}")

    def run_particle_swarm_optimization(self, n_iterations=50, pop_size=10):
        """
        Executa a Otimização por Enxame de Partículas (Particle Swarm Optimization - PSO) para seleção de features.
        O PSO é inspirado no comportamento social de bandos de pássaros ou cardumes de peixes.

        Parâmetros:
        n_iterations (int): Número de iterações (gerações) do algoritmo.
        pop_size (int): Tamanho da população de partículas (soluções).
        """
        print("Executando Particle Swarm Optimization...")
        num_features = self.X.shape[1]
        
        # Inicializa a melhor solução encontrada até agora aleatoriamente
        best_features = np.random.randint(0, 2, size=num_features)
        best_metric = self._evaluate_features(list(self.X.columns[best_features.astype(bool)]))

        # Inicializa as partículas (vetores binários)
        positions = np.random.randint(0, 2, size=(pop_size, num_features)) # Posições das partículas
        velocities = np.zeros((pop_size, num_features)) # Velocidades das partículas
        pbest_positions = positions.copy() # Melhor posição pessoal (pbest) de cada partícula
        pbest_metrics = np.array([self._evaluate_features(list(self.X.columns[p.astype(bool)])) for p in positions]) # Métrica do pbest
        gbest_position = pbest_positions[np.argmax(pbest_metrics)] # Melhor posição global (gbest) encontrada pelo enxame
        gbest_metric = np.max(pbest_metrics) # Métrica do gbest

        c1 = 2.0 # c1: Coeficiente cognitivo (atração à melhor posição pessoal)
        c2 = 2.0 # c2: Coeficiente social (atração à melhor posição global)
        w = 0.9 # w: Peso de inércia (influência da velocidade anterior)

        for iteration in range(n_iterations):
            for i in range(pop_size):
                # Atualiza a velocidade da partícula
                r1, r2 = np.random.rand(num_features), np.random.rand(num_features) # Números aleatórios para estocasticidade
                velocities[i] = w * velocities[i] + \
                                c1 * r1 * (pbest_positions[i] - positions[i]) + \
                                c2 * r2 * (gbest_position - positions[i])

                # Aplica a função Sigmoid para converter a velocidade contínua em uma decisão binária
                s_function = 1 / (1 + np.exp(-velocities[i]))
                positions[i] = (s_function > np.random.rand(num_features)).astype(int) # Atualiza a posição da partícula

                current_features = list(self.X.columns[positions[i].astype(bool)])
                current_metric = self._evaluate_features(current_features)

                # Atualiza a melhor posição pessoal (pbest) se a métrica atual for melhor
                if current_metric > pbest_metrics[i]:
                    pbest_metrics[i] = current_metric
                    pbest_positions[i] = positions[i]

                # Atualiza a melhor posição global (gbest) se a métrica atual for melhor
                if current_metric > gbest_metric:
                    gbest_metric = current_metric
                    gbest_position = positions[i]

            if iteration % 10 == 0:
                print(f"  Iteração {iteration}/{n_iterations}, Melhor Métrica: {gbest_metric:.4f}")

        selected_features = list(self.X.columns[gbest_position.astype(bool)])
        num_features = len(selected_features)
        self.results.append({
            "Técnica": "Particle Swarm Optimization",
            "Valor da Métrica": gbest_metric,
            "Número de Features": num_features,
            "Features Selecionadas": selected_features
        })
        print(f"Particle Swarm Optimization concluído. Métrica: {gbest_metric:.4f}, Features: {num_features}")

    def run_reptile_search_algorithm(self, n_iterations=50, pop_size=10):
        """
        Executa o Algoritmo de Busca de Répteis (Reptile Search Algorithm - RSA) para seleção de features.
        O RSA é inspirado no comportamento de caça e emboscada dos répteis.

        Parâmetros:
        n_iterations (int): Número de iterações (gerações) do algoritmo.
        pop_size (int): Tamanho da população de répteis (soluções).
        """
        print("Executando Reptile Search Algorithm...")
        num_features = self.X.shape[1]
        
        # Inicializa a melhor solução encontrada até agora aleatoriamente
        best_features = np.random.randint(0, 2, size=num_features)
        best_metric = self._evaluate_features(list(self.X.columns[best_features.astype(bool)]))

        # Inicializa a população de répteis (vetores binários)
        population = np.random.randint(0, 2, size=(pop_size, num_features))

        for iteration in range(n_iterations):
            # Encontra a melhor solução na população atual
            fitness = np.array([self._evaluate_features(list(self.X.columns[p.astype(bool)])) for p in population])
            current_best_idx = np.argmax(fitness)
            current_best_solution = population[current_best_idx]
            current_best_metric = fitness[current_best_idx]

            if current_best_metric > best_metric:
                best_metric = current_best_metric
                best_features = current_best_solution

            # Fases de Exploração e Explotação (simplificadas)
            # A exploração ocorre na primeira metade das iterações, a explotação na segunda.
            for i in range(pop_size):
                if iteration < n_iterations / 2: # Fase de Exploração (comportamento de busca)
                    # Caminhada aleatória em torno da melhor solução atual
                    new_position = current_best_solution + np.random.normal(0, 0.1, num_features).astype(int)
                    new_position = np.clip(new_position, 0, 1) # Garante que as features permaneçam binárias
                else: # Fase de Explotação (comportamento de emboscada)
                    # Move-se em direção à melhor solução global
                    new_position = population[i] + np.random.rand() * (best_features - population[i])
                    # Aplica a função Sigmoid para converter a posição contínua em uma decisão binária
                    new_position = (1 / (1 + np.exp(-new_position)) > np.random.rand(num_features)).astype(int)

                current_features = list(self.X.columns[new_position.astype(bool)])
                current_metric = self._evaluate_features(current_features)

                # Se a nova posição for melhor, atualiza a posição do réptil
                if current_metric > self._evaluate_features(list(self.X.columns[population[i].astype(bool)])):
                    population[i] = new_position

            if iteration % 10 == 0:
                print(f"  Iteração {iteration}/{n_iterations}, Melhor Métrica: {best_metric:.4f}")

        selected_features = list(self.X.columns[best_features.astype(bool)])
        num_features = len(selected_features)
        self.results.append({
            "Técnica": "Reptile Search Algorithm",
            "Valor da Métrica": best_metric,
            "Número de Features": num_features,
            "Features Selecionadas": selected_features
        })
        print(f"Reptile Search Algorithm concluído. Métrica: {best_metric:.4f}, Features: {num_features}")

    def run_sine_cosine_algorithm(self, n_iterations=50, pop_size=10):
        """
        Executa o Algoritmo Seno-Cosseno (Sine Cosine Algorithm - SCA) para seleção de features.
        O SCA é inspirado nas funções trigonométricas seno e cosseno para criar flutuações
        que permitem a exploração e explotação do espaço de busca.

        Parâmetros:
        n_iterations (int): Número de iterações (gerações) do algoritmo.
        pop_size (int): Tamanho da população de agentes (soluções).
        """
        print("Executando Sine Cosine Algorithm...")
        num_features = self.X.shape[1]
        
        # Inicializa a melhor solução encontrada até agora aleatoriamente
        best_features = np.random.randint(0, 2, size=num_features)
        best_metric = self._evaluate_features(list(self.X.columns[best_features.astype(bool)]))

        # Inicializa a população de agentes (vetores binários)
        population = np.random.randint(0, 2, size=(pop_size, num_features))

        for iteration in range(n_iterations):
            # Atualiza os parâmetros r1, r2, r3, r4
            # r1: Diminui linearmente de 2 para 0, controlando a distância do movimento
            r1 = 2 - iteration * (2 / n_iterations)
            r2 = 2 * np.pi * np.random.rand() # r2: Número aleatório para a fase do seno/cosseno
            r3 = 2 * np.random.rand() # r3: Número aleatório para a amplitude
            r4 = np.random.rand() # r4: Probabilidade de usar seno ou cosseno

            for i in range(pop_size):
                if r4 < 0.5: # Componente Seno (exploração)
                    new_position = population[i] + r1 * np.sin(r2) * np.abs(r3 * best_features - population[i])
                else: # Componente Cosseno (explotação)
                    new_position = population[i] + r1 * np.cos(r2) * np.abs(r3 * best_features - population[i])
                
                # Aplica a função Sigmoid para converter a posição contínua em uma decisão binária
                new_position = (1 / (1 + np.exp(-new_position)) > np.random.rand(num_features)).astype(int)
                new_position = np.clip(new_position, 0, 1) # Garante que as features permaneçam binárias

                current_features = list(self.X.columns[new_position.astype(bool)])
                current_metric = self._evaluate_features(current_features)

                # Se a nova posição for melhor, atualiza a posição do agente
                if current_metric > self._evaluate_features(list(self.X.columns[population[i].astype(bool)])):
                    population[i] = new_position

                # Atualiza a melhor solução global
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_features = new_position

            if iteration % 10 == 0:
                print(f"  Iteração {iteration}/{n_iterations}, Melhor Métrica: {best_metric:.4f}")

        selected_features = list(self.X.columns[best_features.astype(bool)])
        num_features = len(selected_features)
        self.results.append({
            "Técnica": "Sine Cosine Algorithm",
            "Valor da Métrica": best_metric,
            "Número de Features": num_features,
            "Features Selecionadas": selected_features
        })
        print(f"Sine Cosine Algorithm concluído. Métrica: {best_metric:.4f}, Features: {num_features}")

    def run_whale_optimization_algorithm(self, n_iterations=50, pop_size=10):
        """
        Executa o Algoritmo de Otimização da Baleia (Whale Optimization Algorithm - WOA) para seleção de features.
        O WOA é inspirado no comportamento de caça em bolha das baleias jubarte.

        Parâmetros:
        n_iterations (int): Número de iterações (gerações) do algoritmo.
        pop_size (int): Tamanho da população de baleias (soluções).
        """
        print("Executando Whale Optimization Algorithm...")
        num_features = self.X.shape[1]
        
        # Inicializa a melhor solução encontrada até agora aleatoriamente
        best_features = np.random.randint(0, 2, size=num_features)
        best_metric = self._evaluate_features(list(self.X.columns[best_features.astype(bool)]))

        # Inicializa a população de baleias (vetores binários)
        whales = np.random.randint(0, 2, size=(pop_size, num_features))

        for iteration in range(n_iterations):
            # Encontra a melhor baleia (presa) na população atual
            fitness = np.array([self._evaluate_features(list(self.X.columns[w.astype(bool)])) for w in whales])
            best_whale_idx = np.argmax(fitness)
            best_whale = whales[best_whale_idx]

            if fitness[best_whale_idx] > best_metric:
                best_metric = fitness[best_whale_idx]
                best_features = best_whale

            # Atualiza os parâmetros 'a', 'C', 'l', 'p'
            # 'a': Diminui linearmente de 2 para 0, controlando a exploração vs. explotação
            a = 2 - iteration * (2 / n_iterations)
            # 'l': Número aleatório entre -1 e 1, para definir a forma espiral
            l = (a - 1) * np.random.rand() + 1
            # 'p': Probabilidade de escolher o mecanismo de encurralamento ou espiral
            p = np.random.rand()

            for i in range(pop_size):
                if p < 0.5: # Mecanismo de encurralamento (shrinking encircling mechanism)
                    A = 2 * a * np.random.rand() - a # Fator de ataque
                    C = 2 * np.random.rand() # Fator de exploração
                    D = np.abs(C * best_whale - whales[i]) # Distância da baleia atual para a melhor
                    new_position = best_whale - A * D # Nova posição baseada na melhor baleia
                else: # Caminho em espiral (spiral-shaped path)
                    D_prime = np.abs(best_whale - whales[i]) # Distância da baleia atual para a melhor
                    new_position = D_prime * np.exp(l) * np.cos(2 * np.pi * l) + best_whale # Movimento em espiral
                
                # Aplica a função Sigmoid para converter a posição contínua em uma decisão binária
                new_position = (1 / (1 + np.exp(-new_position)) > np.random.rand(num_features)).astype(int)
                new_position = np.clip(new_position, 0, 1) # Garante que as features permaneçam binárias

                current_features = list(self.X.columns[new_position.astype(bool)])
                current_metric = self._evaluate_features(current_features)

                # Se a nova posição for melhor, atualiza a posição da baleia
                if current_metric > self._evaluate_features(list(self.X.columns[whales[i].astype(bool)])):
                    whales[i] = new_position

            if iteration % 10 == 0:
                print(f"  Iteração {iteration}/{n_iterations}, Melhor Métrica: {best_metric:.4f}")

        selected_features = list(self.X.columns[best_features.astype(bool)])
        num_features = len(selected_features)
        self.results.append({
            "Técnica": "Whale Optimization Algorithm",
            "Valor da Métrica": best_metric,
            "Número de Features": num_features,
            "Features Selecionadas": selected_features
        })
        print(f"Whale Optimization Algorithm concluído. Métrica: {best_metric:.4f}, Features: {num_features}")


if __name__ == "__main__":
    # Exemplo de uso com dataset dummy
    np.random.seed(42) # Semente para reprodutibilidade
    # Cria um DataFrame dummy com 100 amostras e 10 features
    X_dummy = pd.DataFrame(np.random.rand(100, 10), columns=[f"feature_{i}" for i in range(10)])
    # Cria uma série dummy para a variável alvo (binária)
    y_dummy = pd.Series(np.random.randint(0, 2, 100))

    target_metric = 0.75 # Exemplo de valor de métrica de referência para o cálculo do score composto
    target_features = 15 # Exemplo de número de features de referência para o cálculo do score composto

    # Inicializa a classe FeatureSelection com o dataset dummy e a métrica KS
    fs = FeatureSelection(X_dummy, y_dummy, metric="ks")

    # Executa cada técnica de seleção de features
    # Os parâmetros n_iterations e pop_size foram reduzidos para agilizar a execução do exemplo.
    # Em um cenário real, esses valores devem ser maiores e otimizados para cada problema.
    fs.run_boruta()
    fs.run_binary_bat_algorithm(n_iterations=5, pop_size=3)
    fs.run_cuckoo_search(n_iterations=5, pop_size=3)
    fs.run_equilibrium_optimizer(n_iterations=5, pop_size=3)
    fs.run_genetic_algorithm(n_iterations=5, pop_size=3)
    fs.run_gravitational_search_algorithm(n_iterations=5, pop_size=3)
    fs.run_grey_wolf_optimizer(n_iterations=5, pop_size=3)
    fs.run_harmony_search(n_iterations=5, pop_size=3)
    fs.run_memetic_algorithm(n_iterations=5, pop_size=3)
    fs.run_particle_swarm_optimization(n_iterations=5, pop_size=3)
    fs.run_reptile_search_algorithm(n_iterations=5, pop_size=3)
    fs.run_sine_cosine_algorithm(n_iterations=5, pop_size=3)
    fs.run_whale_optimization_algorithm(n_iterations=5, pop_size=3)

    # Obtém o ranking final das técnicas
    ranking_df = fs.get_ranking(target_metric, target_features)
    print("\nRanking Final:")
    print(ranking_df.to_string())


