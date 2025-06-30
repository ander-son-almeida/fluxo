import pandas as pd
import numpy as np

# Number of customers and months
num_customers = 1000
num_months = 12

# Generate customer IDs
customer_ids = [f'C{i}' for i in range(1, num_customers + 1)]

# Generate monthly data
data = []
for customer_id in customer_ids:
    for month in range(1, num_months + 1):
        data.append([customer_id, month])

df = pd.DataFrame(data, columns=['customer_id', 'month'])

# Generate core features
np.random.seed(42)
df['pontuacao_credito'] = np.random.randint(300, 850, size=len(df))
df['renda'] = np.random.uniform(1000, 10000, size=len(df))
df['valor_emprestimo'] = np.random.uniform(500, 50000, size=len(df))
df['historico_pagamento'] = np.random.choice([0, 1, 2], size=len(df), p=[0.7, 0.2, 0.1]) # 0: em dia, 1: atrasado, 2: muito atrasado
df['anos_emprego'] = np.random.randint(0, 30, size=len(df))

# Generate additional features (at least 45 more)
for i in range(1, 46):
    df[f'feature_{i}'] = np.random.rand(len(df)) * 100

# Generate 'default' column with a realistic imbalanced proportion
default_probability = 0.02 # 2% default rate (more realistic)
df['inadimplencia'] = np.random.choice([0, 1], size=len(df), p=[1 - default_probability, default_probability])

# Simulate changes leading to default (only for actual defaulters)
# Ensure that only customers who are marked as default=1 in the current month
# and were not default in the previous month (if applicable) have their features changed.
for i in range(1, len(df)):
    if df.loc[i, 'inadimplencia'] == 1 and df.loc[i-1, 'inadimplencia'] == 0 and df.loc[i, 'customer_id'] == df.loc[i-1, 'customer_id']:
        df.loc[i, 'pontuacao_credito'] -= np.random.randint(50, 150)
        df.loc[i, 'renda'] *= np.random.uniform(0.5, 0.8)
        df.loc[i, 'historico_pagamento'] = np.random.choice([1, 2], p=[0.5, 0.5])
        # Also affect some new features
        for j in range(1, 10): # Affect 10 random new features
            if f'feature_{j}' in df.columns:
                df.loc[i, f'feature_{j}'] *= np.random.uniform(0.7, 0.9)

# Save to CSV
df.to_csv('bank_customer_data.csv', index=False)

print('Dataset created successfully with more features and adjusted default rate!')


