from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import base64
from io import BytesIO
import numpy as np
import pandas as pd


import joblib

from fastapi.middleware.cors import CORSMiddleware

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------
# Load trained models and scaler
# -----------------------------------------------------
logreg = joblib.load("logreg_best.pkl")
dtree = joblib.load("decision_tree.pkl")
rf = joblib.load("random_forest.pkl")
svm = joblib.load("svm_rbf.pkl")
scaler = joblib.load("scaler.pkl")

# Map for model selection
models = {
    "logreg": logreg,
    "decision_tree": dtree,
    "random_forest": rf,
    "svm": svm,
}

# indices of continuous features in the order:
# [distance, pressure, hrv, sugar_level, spo2, accelerometer]
continuous_idx = [0, 2, 3, 4]

# -----------------------------------------------------
# FastAPI app
# -----------------------------------------------------
app = FastAPI(
    title="Fall Prediction API",
    description="FastAPI backend for cStick fall prediction with multiple ML models.",
    version="1.0.0"
)

# Allow frontend (browser) access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# Pydantic models
# -----------------------------------------------------
class SinglePredictionRequest(BaseModel):
    model_name: str
    distance: float
    pressure: float
    hrv: float
    sugar_level: float
    spo2: float
    accelerometer: float

class BatchPredictionRequest(BaseModel):
    inputs: List[SinglePredictionRequest]

# -----------------------------------------------------
# Helper: load full dataset
# -----------------------------------------------------
def load_dataset():
    df = pd.read_csv("cStick.csv")
    df.columns = df.columns.str.strip()
    X = df.drop(columns=["Decision"])
    y = df["Decision"]
    return df, X, y

# =====================================================
# ENDPOINTS
# =====================================================

# 1) HEALTH CHECK -------------------------------------------------------
@app.get("/health")
def health():
    """
    Basic health check endpoint.
    """
    return {"status": "OK", "message": "API is running"}

# 2) LIST AVAILABLE MODELS ----------------------------------------------
@app.get("/models")
def get_models():
    """
    Returns list of available ML models.
    """
    return {"available_models": list(models.keys())}

# 3) DATASET PREVIEW -----------------------------------------------------
@app.get("/dataset/preview")
def dataset_preview():
    """
    Returns the first 5 rows of the dataset.
    """
    df = pd.read_csv("cStick.csv")
    df.columns = df.columns.str.strip()
    return df.head(5).to_dict(orient="records")

# 4) SINGLE PREDICTION WITH SELECTED MODEL ------------------------------
@app.post("/predict/model")
def predict_model(payload: SinglePredictionRequest):
    """
    Uses the selected model to make a single prediction.
    """
    if payload.model_name not in models:
        return {"error": f"Model '{payload.model_name}' not found."}

    model = models[payload.model_name]

    x = np.array([[payload.distance,
                   payload.pressure,
                   payload.hrv,
                   payload.sugar_level,
                   payload.spo2,
                   payload.accelerometer]])

    # scale continuous features
    x_scaled = x.copy()
    x_scaled[:, continuous_idx] = scaler.transform(x[:, continuous_idx])

    pred = int(model.predict(x_scaled)[0])
    proba = model.predict_proba(x_scaled)[0].tolist()

    labels = ["No Fall", "Fall Predicted", "Fall Detected"]

    return {
        "model_used": payload.model_name,
        "prediction_raw": pred,
        "prediction_label": labels[pred],
        "probabilities": {
            "No Fall": proba[0],
            "Fall Predicted": proba[1],
            "Fall Detected": proba[2],
        },
    }

# 5) BATCH PREDICTION ----------------------------------------------------
@app.post("/predict/batch")
def predict_batch(batch: BatchPredictionRequest):
    """
    Performs batch prediction for multiple samples.
    Each input includes a model_name and sensor values.
    """
    results = []

    for item in batch.inputs:
        if item.model_name not in models:
            results.append({"error": f"Model '{item.model_name}' not found."})
            continue

        model = models[item.model_name]

        x = np.array([[item.distance,
                       item.pressure,
                       item.hrv,
                       item.sugar_level,
                       item.spo2,
                       item.accelerometer]])

        x_scaled = x.copy()
        x_scaled[:, continuous_idx] = scaler.transform(x[:, continuous_idx])

        pred = int(model.predict(x_scaled)[0])
        proba = model.predict_proba(x_scaled)[0].tolist()

        labels = ["No Fall", "Fall Predicted", "Fall Detected"]

        results.append({
            "model_used": item.model_name,
            "prediction_raw": pred,
            "prediction_label": labels[pred],
            "probabilities": {
                "No Fall": proba[0],
                "Fall Predicted": proba[1],
                "Fall Detected": proba[2],
            },
        })

    return {"results": results}

# 6) MODEL ACCURACY (USING LOGISTIC REGRESSION BY DEFAULT) --------------
@app.get("/model/accuracy")
def model_accuracy():
    """
    Computes accuracy of the default model (logreg) on the full dataset.
    """
    _, X, y = load_dataset()

    X_scaled = X.copy()
    X_scaled.iloc[:, continuous_idx] = scaler.transform(X.iloc[:, continuous_idx])

    preds = logreg.predict(X_scaled)
    acc = accuracy_score(y, preds)

    return {"model": "logreg", "accuracy": acc}

# 7) MODEL METRICS (USING LOGISTIC REGRESSION BY DEFAULT) ---------------
@app.get("/model/metrics")
def model_metrics():

    # Load dataset
    _, X, y = load_dataset()

    # Scale continuous features
    X_scaled = X.copy()
    X_scaled.iloc[:, continuous_idx] = scaler.transform(X.iloc[:, continuous_idx])

    # Predict using Logistic Regression (default evaluation model)
    preds = logreg.predict(X_scaled)

    # Compute confusion matrix
    cm = confusion_matrix(y, preds)

    # Build confusion matrix table with headers
    cm_table = [
        ["", "Pred 0", "Pred 1", "Pred 2"],
        ["Actual 0", int(cm[0][0]), int(cm[0][1]), int(cm[0][2])],
        ["Actual 1", int(cm[1][0]), int(cm[1][1]), int(cm[1][2])],
        ["Actual 2", int(cm[2][0]), int(cm[2][1]), int(cm[2][2])]
    ]

    # Build clean classification report
    cr = classification_report(y, preds, output_dict=True)

    cr_table = [
        {
            "class": "No Fall (0)",
            "precision": round(cr["0"]["precision"], 3),
            "recall": round(cr["0"]["recall"], 3),
            "f1_score": round(cr["0"]["f1-score"], 3),
            "support": int(cr["0"]["support"])
        },
        {
            "class": "Fall Predicted (1)",
            "precision": round(cr["1"]["precision"], 3),
            "recall": round(cr["1"]["recall"], 3),
            "f1_score": round(cr["1"]["f1-score"], 3),
            "support": int(cr["1"]["support"])
        },
        {
            "class": "Fall Detected (2)",
            "precision": round(cr["2"]["precision"], 3),
            "recall": round(cr["2"]["recall"], 3),
            "f1_score": round(cr["2"]["f1-score"], 3),
            "support": int(cr["2"]["support"])
        }
    ]

    accuracy = accuracy_score(y, preds)

    return {
        "model": "logreg",
        "accuracy": accuracy,
        "confusion_matrix_table": cm_table,
        "classification_report_table": cr_table
    }


# 8) MODEL RETRAIN (ONLY LOGREG HERE) -----------------------------------
@app.post("/model/retrain")
def model_retrain():
    """
    Retrains the logistic regression model using the current CSV dataset.
    Updates the saved model and scaler.
    """
    global logreg, scaler, models

    df, X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # new scaler
    new_scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled.iloc[:, continuous_idx] = new_scaler.fit_transform(
        X_train.iloc[:, continuous_idx]
    )
    X_test_scaled.iloc[:, continuous_idx] = new_scaler.transform(
        X_test.iloc[:, continuous_idx]
    )

    # retrain logistic regression
    new_logreg = LogisticRegression(max_iter=500, multi_class="multinomial")
    new_logreg.fit(X_train_scaled, y_train)

    # evaluate
    test_preds = new_logreg.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)

    # save updated model + scaler
    joblib.dump(new_logreg, "logreg_best.pkl")
    joblib.dump(new_scaler, "scaler.pkl")

    # update globals
    logreg = new_logreg
    scaler = new_scaler
    models["logreg"] = new_logreg

    return {
        "message": "Logistic Regression model retrained and updated.",
        "new_test_accuracy": test_acc,
    }



@app.get("/model/roc/{model_name}")
def get_roc_curve(model_name: str):

    if model_name not in models:
        return {"error": f"Model '{model_name}' not found."}

    model = models[model_name]

    _, X, y = load_dataset()
    X_scaled = X.copy()
    X_scaled.iloc[:, continuous_idx] = scaler.transform(X.iloc[:, continuous_idx])

    y_score = model.predict_proba(X_scaled)
    y_bin = label_binarize(y, classes=[0,1,2])

    fpr = {}
    tpr = {}
    roc_auc = {}

    plt.figure(figsize=(6,4))
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC={roc_auc[i]:.2f})")

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"roc_curve": img_base64}


@app.get("/model/pr/{model_name}")
def get_pr_curve(model_name: str):

    if model_name not in models:
        return {"error": f"Model '{model_name}' not found."}

    model = models[model_name]

    _, X, y = load_dataset()
    X_scaled = X.copy()
    X_scaled.iloc[:, continuous_idx] = scaler.transform(X.iloc[:, continuous_idx])

    y_score = model.predict_proba(X_scaled)
    y_bin = label_binarize(y, classes=[0,1,2])

    plt.figure(figsize=(6,4))
    for i in range(3):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f"Class {i}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve - {model_name}")
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return {"pr_curve": img_base64}


@app.get("/model/compare")
def compare_models():

    _, X, y = load_dataset()
    X_scaled = X.copy()
    X_scaled.iloc[:, continuous_idx] = scaler.transform(X.iloc[:, continuous_idx])

    results = []

    for name, model in models.items():
        preds = model.predict(X_scaled)
        acc = accuracy_score(y, preds)
        results.append({
            "model": name,
            "accuracy": acc
        })

    return {"comparison": results}
