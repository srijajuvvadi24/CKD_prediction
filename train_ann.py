import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- LOAD DATASET ----------------

df = pd.read_csv("kidney_disease.csv")

if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

df = df[['age', 'bp', 'sg', 'al', 'su', 'hemo', 'classification']]

# Missing values
df = df.fillna(df.mode().iloc[0])

# Clean target
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
X = df.drop('classification', axis=1)
y = df['classification']

# ---------------- SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- SCALING ----------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- ANN MODEL ----------------

model = Sequential()

model.add(Dense(32, activation='relu', input_shape=(6,)))
model.add(Dropout(0.4))

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ---------------- TRAIN ----------------

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=25,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# ---------------- EVALUATE ----------------

loss, acc = model.evaluate(X_test, y_test)

print("\nANN Accuracy:",
      round(acc * 100, 2), "%")

# Predictions
y_pred_prob = model.predict(X_test)

y_pred = (y_pred_prob > 0.5).astype(int)

# Reports
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ---------------- SAVE ----------------

model.save("ann_model.h5")

pickle.dump(
    scaler,
    open("scaler.pkl", "wb")
)

print("\n✅ ANN Model Saved")
