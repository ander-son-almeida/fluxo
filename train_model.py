import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib

# Load the dataset
df = pd.read_csv("bank_customer_data.csv")

# Drop customer_id and month for training
X = df.drop(["customer_id", "month", "inadimplencia"], axis=1)
y = df["inadimplencia"]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Handle imbalanced data with SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train LightGBM model
lgbm = lgb.LGBMClassifier(random_state=42, is_unbalance=True) # is_unbalance=True for handling imbalanced data
lgbm.fit(X_train_res, y_train_res)

# Save the model, scaler, X_test, and y_test
joblib.dump(lgbm, "lightgbm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

print("Model trained and saved successfully!")


