Elderly Fall Prediction using IoMT Sensor Data

A machine learning–based fall prediction system for elderly care, built using the cStick IoMT dataset and deployed via FastAPI for real-time and batch inference.

This project demonstrates a complete ML workflow—from data preprocessing and model evaluation to API-based deployment—aimed at improving elderly safety through early fall risk detection.

📌 Project Overview

Falls are a major health risk for elderly individuals. This project leverages IoMT sensor data collected from a smart cane (cStick) to classify user states into:

Safe

Warning (Fall Predicted)

Fall Detected

The final model is integrated into a FastAPI backend, enabling real-time predictions that can be consumed by IoT devices, mobile apps, or healthcare dashboards.

🧠 Dataset

Name: cStick Elderly Fall Prediction Dataset

Source: Kaggle

Format: CSV

Size: 2,039 rows × 7 columns

Features
Feature	Description
Distance	Distance from obstacles (cm)
Pressure	Grip strength (0 = low, 1 = medium, 2 = high)
HRV	Heart rate (BPM)
Sugar Level	Blood glucose (mg/dL)
SpO₂	Blood oxygen saturation (%)
Accelerometer	Sudden movement indicator (0/1)
Decision	Target label (0, 1, 2)
Target Classes
Label	Meaning
0	Safe / No fall
1	Warning / Fall predicted
2	Fall detected
🔬 Methodology

Data Exploration

Distribution analysis

Outlier inspection (retained due to clinical relevance)

Class balance verification

Preprocessing

Standardization of continuous features

No missing-value imputation required

Stratified 80/20 train–test split

Model Training & Comparison

Logistic Regression (Multinomial)

Decision Tree

Random Forest

SVM (RBF Kernel)

Model Selection

Logistic Regression selected for deployment due to:

100% accuracy

Low computational cost

High interpretability

Suitability for real-time IoMT systems

📊 Results

Accuracy: 100%

Precision / Recall / F1-score: 1.00 for all classes

Confusion Matrix: Perfect classification across all test samples

The dataset exhibits deterministic separability, enabling highly reliable predictions.

🚀 FastAPI Backend

The trained model is deployed using FastAPI, providing RESTful endpoints for prediction and evaluation.

Key API Endpoints
Health & Docs

GET /health – Server status

GET /docs – Interactive Swagger UI

Predictions

POST /predict – Single sample prediction

POST /predict/batch – Batch predictions

Model Info

GET /model/accuracy

GET /model/metrics

GET /model/roc

GET /model/pr

Advanced Operations

POST /model/retrain

GET /model/compare

Sample Request
POST /predict
{
  "Distance": 12.5,
  "Pressure": 2,
  "HRV": 105.4,
  "Sugar level": 25.3,
  "SpO2": 72.1,
  "Accelerometer": 1
}

Sample Response
{
  "prediction": 2,
  "probabilities": {
    "0": 0.00001,
    "1": 0.00043,
    "2": 0.99956
  }
}

🛠️ Tech Stack

Python

scikit-learn

FastAPI

Pydantic

NumPy / Pandas

Uvicorn

⚠️ Limitations

Dataset is small and highly clean

No temporal modeling (each sample treated independently)

Single-device data source

Real-world noise and sensor drift not included

🔮 Future Work

Time-series modeling (LSTM, Transformers)

Real-world IoMT data integration

Edge-device deployment

Expanded dataset with multiple users and environments

Alert systems for caregivers and emergency services
