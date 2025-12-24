import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ----------------------------------------------------
# 1. Load dataset
# ----------------------------------------------------
df = pd.read_csv("cStick.csv")
df.columns = df.columns.str.strip()

X = df.drop(columns=["Decision"])
y = df["Decision"]

continuous_features = ["Distance", "HRV", "Sugar level", "SpO2"]

# ----------------------------------------------------
# 2. Train-test split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------
# 3. Standardize continuous features
# ----------------------------------------------------
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])

# ----------------------------------------------------
# 4. Train ALL ML models
# ----------------------------------------------------
logreg = LogisticRegression(max_iter=500, multi_class="multinomial")
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(n_estimators=200, random_state=42)
svm_rbf = SVC(probability=True, random_state=42)

models = {
    "logreg": logreg,
    "decision_tree": decision_tree,
    "random_forest": random_forest,
    "svm_rbf": svm_rbf
}

# ----------------------------------------------------
# 5. Fit and evaluate each model
# ----------------------------------------------------
for name, model in models.items():
    print(f"\n======= Training {name} =======")
    
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

# ----------------------------------------------------
# 6. Save ALL models
# ----------------------------------------------------
print("\nSaving all models...")

joblib.dump(logreg, "logreg_best.pkl")
joblib.dump(decision_tree, "decision_tree.pkl")
joblib.dump(random_forest, "random_forest.pkl")
joblib.dump(svm_rbf, "svm_rbf.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Models saved successfully!")
