import pandas as pd
import re

# Função para extrair a chave única do arquivo (exemplo - ajuste conforme seus dados)
def extrair_chave(caminho):
    nome_arquivo = caminho.split('\\')[-1]  # Pega o nome do arquivo
    # Remove extensão e versão (ajuste este regex conforme seu padrão de nomes)
    chave = re.sub(r'(_v\d+)?\.\w+$', '', nome_arquivo)  
    return chave

# Adiciona coluna de chave ao DataFrame
df['chave'] = df['caminho'].apply(extrair_chave)

# Função para selecionar o melhor arquivo de cada grupo
def selecionar_melhor(grupo):
    # Ordena o grupo: Word primeiro, depois maior versão
    grupo_ordenado = grupo.sort_values(by=['tipo', 'versão'], 
                                     ascending=[True, False])
    return grupo_ordenado.iloc[0]  # Pega o primeiro (melhor) registro

# Aplica a seleção para cada grupo de arquivos
df_final = df.groupby('chave', group_keys=False).apply(selecionar_melhor)

# Remove a coluna auxiliar 'chave' se não for mais necessária
df_final = df_final.drop(columns=['chave'])

# Resultado final
print(df_final[['tipo', 'versão', 'caminho']])