import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

# ---------------- LOAD DATASET ----------------

df = pd.read_csv("kidney_disease.csv")

# Drop id column
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Select important features
df = df[['age', 'bp', 'sg', 'al', 'su', 'hemo', 'classification']]

# Handle missing values
df = df.fillna(df.mode().iloc[0])

# Clean target column
df['classification'] = (
    df['classification']
    .astype(str)
    .str.strip()
    .str.lower()
)

df['classification'] = df['classification'].map({
    'ckd': 1,
    'notckd': 0
})

df.dropna(subset=['classification'], inplace=True)

# Features & target
X = df[['age', 'bp', 'sg', 'al', 'su', 'hemo']]
y = df['classification']

# ---------------- TRAIN TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- RANDOM FOREST ----------------

base_model = RandomForestClassifier(
    n_estimators=80,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42
)

# Probability calibration
model = CalibratedClassifierCV(
    base_model,
    method='sigmoid',
    cv=5
)

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nRandom Forest Accuracy:", round(accuracy * 100, 2), "%")

# Cross Validation
cv_scores = cross_val_score(base_model, X, y, cv=5)

print("Cross Validation Mean Accuracy:",
      round(cv_scores.mean() * 100, 2), "%")

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# Train base model separately
base_model.fit(X_train, y_train)

# ---------------- FEATURE IMPORTANCE ----------------

features = X.columns
importance = base_model.feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(features, importance)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance")

plt.show()

# ---------------- SAVE MODEL ----------------

pickle.dump({
    "calibrated_model": model,
    "base_model": base_model
}, open("ckd_model.pkl", "wb"))

print("\n✅ Random Forest Model Saved")
