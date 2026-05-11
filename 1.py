##############################
# Telco Customer Churn Project
##############################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

pd.set_option("display.max_columns", None)

############################################
# 1️⃣ DATASET YÜKLEME
############################################

df = pd.read_csv("Telco-Customer-Churn.csv")

print("Initial Shape:", df.shape)
print(df.head(10))

############################################
# 2️⃣ DATA TYPES & MISSING VALUES
############################################

print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# TotalCharges numeric olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Median ile doldur
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

############################################
# 3️⃣ TARGET DÖNÜŞÜMÜ
############################################

df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

############################################
# 4️⃣ customerID KALDIR
############################################

df.drop("customerID", axis=1, inplace=True)

############################################
# 5️⃣ HEDEF DAĞILIMI
############################################

sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

print("Churn Ratio:\n", df["Churn"].value_counts(normalize=True))

############################################
# 6️⃣ NUMERİK & KATEGORİK AYRIMI
############################################

cat_cols = [col for col in df.columns if df[col].dtype == "O"]
num_cols = [col for col in df.columns if df[col].dtype != "O"]

print("Categorical Columns:", cat_cols)
print("Numerical Columns:", num_cols)

############################################
# 7️⃣ KORELASYON ANALİZİ
############################################

plt.figure(figsize=(10,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="magma")
plt.title("Correlation Matrix")
plt.show()

############################################
# 8️⃣ ENCODING
############################################

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

############################################
# 9️⃣ FEATURE & TARGET AYIRIMI
############################################

y = df_encoded["Churn"]
X = df_encoded.drop("Churn", axis=1)

print("X shape:", X.shape)
print("y shape:", y.shape)

############################################
# 🔟 TRAIN - TEST SPLIT
############################################
############################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

############################################
# 1️⃣1️⃣ BASELINE MODEL (CatBoost)
############################################

model = CatBoostClassifier(verbose=False, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

############################################
# 1️⃣2️⃣ MODEL PERFORMANCE
############################################

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("F1:", round(f1_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_pred), 4))