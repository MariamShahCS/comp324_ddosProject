# class distribution graph
'''
os.makedirs("reports", exist_ok=True)
plt.figure(figsize=(6,4)) # (width, height) of chart
counts = y.value_counts()
plt.bar(counts.index.astype(str), counts.values, edgecolor='black')
plt.xticks([0,1], labels=['Benign (0)', 'DDoS (1)'], rotation=0)
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Dataset Class Distribution")
plt.tight_layout()
plt.savefig("reports/class_distribution.png")
plt.show()'''

# joblib stuff:
# to eval w/o retraining in new file:
#   import joblib & pandas
# -> rf_model = joblib.load("ddos_detector.joblib")
# -> load new data to eval: data = pd.DataFrame({...})
# -> evals = rf_model.predict(data), print(evals) # EX output: [0 1] 

# model_tester.py
# Compare Random Forest, Logistic Regression, and Neural Net
# using ROC / AUC on the same test data.

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .features import FEATURE_NAMES  # names for the columns :contentReference[oaicite:2]{index=2}

# --------------------------------------------------------------------
# BASIC SETTINGS
# --------------------------------------------------------------------

# These are the dataset names used everywhere else in your project. :contentReference[oaicite:3]{index=3}
ATTACK_TYPES = ["DNS", "LDAP", "MSSQL", "NETBIOS", "NTP",
                "SNMP", "SSDP", "SYN", "TFTP", "UDP", "UDPLAG"]

# Where to find the training data joblib files created by prep_training_data.py
JOBLIB_FILES = {
    key: f"data/joblibs/training_{key}_data.joblib"
    for key in ATTACK_TYPES
}

# Which models we want to compare
MODEL_TYPES = ["RF", "LR", "NN"]
MODEL_NAMES = {
    "RF": "Random Forest",
    "LR": "Logistic Regression",
    "NN": "Neural Network",
}

SEED_VAL = 42  # fixed seed so the train/test split is repeatable

# --------------------------------------------------------------------
# STEP 1: LOAD AND CLEAN DATA FOR TESTING
# --------------------------------------------------------------------

def load_datasets_for_testing():
    """
    Choose which datasets to use, load their joblib files,
    glue them together into one big X (inputs) and y (labels),
    and clean the numbers a bit.
    """
    print("Available datasets:", ", ".join(ATTACK_TYPES))
    user_input = input(
        "Enter datasets to test on (comma-separated, or 'ALL'): "
    ).upper().strip()

    if user_input == "ALL":
        selected = ATTACK_TYPES
    else:
        pieces = [p.strip() for p in user_input.split(",") if p.strip()]
        selected = []
        for p in pieces:
            if p not in ATTACK_TYPES:
                print(f"[!] WARNING: '{p}' is not a valid dataset key, skipping.")
            else:
                selected.append(p)
        if not selected:
            print("[!] No valid datasets entered, defaulting to ALL.")
            selected = ATTACK_TYPES

    all_X = []
    all_y = []

    for key in selected:
        path = JOBLIB_FILES[key]
        if not os.path.exists(path):
            print(f"[!] WARNING: joblib file {path} not found, skipping.")
            continue
        X_part, y_part = joblib.load(path)  # (features, labels) :contentReference[oaicite:4]{index=4}
        all_X.append(X_part)
        all_y.append(y_part)

    if not all_X:
        raise RuntimeError(
            "No datasets could be loaded. "
            "Make sure you ran prep_training_data.py first."
        )

    # Stitch all parts together into one big table
    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)

    # Clean weird values (same logic as supervised_learning_model.py). :contentReference[oaicite:5]{index=5}
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    X = X.clip(-1e12, 1e12)
    X = X.astype(np.float32)

    # Give columns readable names using FEATURE_NAMES
    name_map = {}
    for col in X.columns:
        if isinstance(col, int) and 0 <= col < len(FEATURE_NAMES):
            name_map[col] = FEATURE_NAMES[col]
        else:
            name_map[col] = f"feature_{col}"
    X = X.rename(columns=name_map)

    print("\nLoaded test data from:", ", ".join(selected))
    print("Total samples:", len(X))
    print("Label counts:\n", y.value_counts())

    return X, y, selected

# --------------------------------------------------------------------
# STEP 2: BUILD MODELS
# --------------------------------------------------------------------

def build_model(model_type):
    """
    Create one of the three model objects,
    using the same settings as in supervised_learning_model.py.
    """
    if model_type == "RF":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=SEED_VAL,
            n_jobs=-1,
        )
    elif model_type == "LR":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=SEED_VAL,
        )
    elif model_type == "NN":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=300,
            learning_rate="adaptive",
            random_state=SEED_VAL,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# --------------------------------------------------------------------
# STEP 3 & 4: TRAIN, GET ROC/AUC, AND COMPARE
# --------------------------------------------------------------------

if __name__ == "__main__":
    # 1) Load data
    X, y, used_datasets = load_datasets_for_testing()

    # 2) Split into train and test once, so all models use the SAME test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED_VAL,
    )

    print("\nTrain size:", X_train.shape[0], " | Test size:", X_test.shape[0])

    # Where we'll store the results for each model
    roc_info = {}     # for ROC curve lines
    metric_info = {}  # for accuracy, precision, recall, F1, AUC

    # 3) Loop over the three model types
    for mtype in MODEL_TYPES:
        print(f"\n=== Training {MODEL_NAMES[mtype]} model ===")

        # LR and NN need scaling; RF does not.
        if mtype in ["LR", "NN"]:
            scaler = StandardScaler()
            X_train_use = scaler.fit_transform(X_train)
            X_test_use = scaler.transform(X_test)
        else:
            scaler = None
            X_train_use = X_train
            X_test_use = X_test

        model = build_model(mtype)
        model.fit(X_train_use, y_train)

        # Predictions (hard 0/1) and probabilities (how sure it's class 1)
        y_pred = model.predict(X_test_use)
        y_prob = model.predict_proba(X_test_use)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # Save metrics for later plotting/summary
        roc_info[mtype] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        metric_info[mtype] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": roc_auc,
        }

        print(f"{MODEL_NAMES[mtype]} results:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-score : {f1:.4f}")
        print(f"  AUC      : {roc_auc:.4f}")

    # 4) Plot all three ROC curves on one figure
    plt.figure(figsize=(8, 6))
    for mtype in MODEL_TYPES:
        info = roc_info[mtype]
        label = f"{MODEL_NAMES[mtype]} (AUC = {info['auc']:.2f})"
        plt.plot(info["fpr"], info["tpr"], label=label)

    # Random-guessing baseline (diagonal line)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier (AUC = 0.50)")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5) Print a simple "winner" using AUC
    best_model = max(metric_info.items(), key=lambda item: item[1]["auc"])
    best_type, best_metrics = best_model
    print("\nBest model by AUC:", MODEL_NAMES[best_type])
    print("AUC:", best_metrics["auc"])

