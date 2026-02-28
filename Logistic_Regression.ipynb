# ==========================================
# Customer Churn Prediction using Logistic Regression
# ==========================================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, classification_report


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("churn_data.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())


# ==========================================
# 2. Data Preprocessing
# ==========================================

# Selecting features (modify according to your dataset columns)
X = df.drop("churn", axis=1)   # Features
y = df["churn"]                # Target variable

# Convert categorical columns if any
X = pd.get_dummies(X, drop_first=True)


# ==========================================
# 3. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# ==========================================
# 4. Feature Scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# 5. Model Training
# ==========================================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ==========================================
# 6. Predictions
# ==========================================

y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)


# ==========================================
# 7. Model Evaluation
# ==========================================

print("\nLog Loss:", log_loss(y_test, y_pred_prob))
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==========================================
# 8. Probability Distribution Visualization
# ==========================================

plt.figure(figsize=(6,4))
plt.hist(y_pred_prob[:,1], bins=20)
plt.title("Predicted Churn Probabilities")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.show()


# ==========================================
# 9. Feature Importance (Coefficient Analysis)
# ==========================================

feature_importance = pd.Series(model.coef_[0], index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(8,5))
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.show()
