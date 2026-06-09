# CKD Prediction System

## Overview
The CKD Prediction System is a machine learning and deep learning based web application used to predict Chronic Kidney Disease (CKD) from patient health parameters.

The project allows users to:
- Enter medical details
- Select prediction models
- Predict CKD status
- View model performance metrics

---

## Technologies Used

- Python
- Flask
- HTML
- CSS
- JavaScript
- Machine Learning
- Deep Learning
- Pickle
- TensorFlow / Keras

---

## Models Used

### Machine Learning Models
- Random Forest
- Decision Tree
- Logistic Regression

### Deep Learning Models
- ANN (Artificial Neural Network)

---

## Dataset

The dataset used contains medical parameters related to kidney disease prediction.

Example attributes:
- Age
- Blood Pressure
- Sugar
- Albumin
- Hemoglobin
- Serum Creatinine
- Diabetes Mellitus
- Hypertension

---

## Project Structure

```bash
CKD_prediction/
│
├── app.py
├── auth.py
├── kidney_disease.csv
├── Testing_CKD_dataset.csv
├── ckd_model.pkl
├── ckd_ml_model.pkl
├── ann_model.h5
├── ckd_dl_model.h5
├── feature_names.pkl
├── label_encoders.pkl
│
├── templates/
├── static/
└── README.md
