import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, os, random, csv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from .features import FEATURE_NAMES

# GLOBAL VARIABLES =========================================
SEED_VAL = 42           # for random seeding/testing reproducability, you hitchhiker
RAND_FLAG = True        # if random set True, elif fixed set False
DATA_FLAG = True        # if using dummyData set False, elif using real data set True
ALL_SETS_FLAG = False   # if using all (real) datasets set True, elif want user input to select datasets set False
ATTACK_TYPES = ["DNS","LDAP","MSSQL","NETBIOS","NTP","SNMP","SSDP","SYN","TFTP","UDP","UDPLAG"]
JOBLIB_FILES = {
                key:f"data/joblibs/training_{key}_data.joblib"
                for key in ATTACK_TYPES
                }

# PATH CHECKS ==============================================
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# FUNCTION DEFINITIONS =====================================
# genDummyData(): generate dummy dataset with super simple indicators
# - returns a dataframe
def genDummyData(n_samples=2000, ddos_ratio=0.3):
    if (RAND_FLAG): rng = np.random.default_rng()
    else:           rng = np.random.default_rng(SEED_VAL)

    # normal vs ddos
    n_ddos      = int(n_samples * ddos_ratio)
    n_normal    = n_samples - n_ddos

    # normal traffic
    #   loc=60 & scale=10 (most vals between 50 and 70)
    #   typical normal = tens of packets per sec 
    normal_packets_per_sec  = rng.normal(loc=60, scale=10, size=n_normal)
    normal_bytes_per_sec    = rng.normal(loc=25000, scale=4000, size=n_normal)
    normal_duration         = rng.normal(loc=2.0, scale=0.4, size=n_normal)

    # ddos traffic
    #   more variation -> bigger scale
    #   typical ddos -> bigger numbers/packet flooding...
    ddos_packets_per_sec    = rng.normal(loc=400, scale=60, size=n_ddos)
    ddos_bytes_per_sec      = rng.normal(loc=150000, scale=25000, size=n_ddos)
    ddos_duration           = rng.normal(loc=0.2, scale=0.4, size=n_ddos)

    # stacks
    packets_per_sec = np.concatenate([normal_packets_per_sec, ddos_packets_per_sec])
    bytes_per_sec   = np.concatenate([normal_bytes_per_sec, ddos_bytes_per_sec])
    duration_sec    = np.concatenate([normal_duration, ddos_duration])

    # labels (0=normal, 1=ddos)
    labels = np.array([0]*n_normal + [1]*n_ddos)

    # dataframe
    df = pd.DataFrame({ "packets_per_sec":  packets_per_sec,
                        "bytes_per_sec":    bytes_per_sec,
                        "duration_sec":     duration_sec,
                        "label":            labels 
                        })
    
    #shuffle
    if (RAND_FLAG): df = df.sample(frac=1.0).reset_index(drop=True)
    else:           df = df.sample(frac=1.0, random_state=SEED_VAL).reset_index(drop=True)
    
    return df

# measureRandSpread(): measure & print random spread accuracy, only rand trials
# - enter parameter # of trials you'd like to measure
def measureRandSpread(trials):
    acc_scores = []
    recall_scores = []
    for i in range(trials):
        data = genDummyData()
        if (i<3): print("\n===Sample of data from trial ",i,"===\n",data.head())
        X = data.drop(columns=["label"])
        y = data["label"]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)
        model = RandomForestClassifier(n_estimators=200,class_weight="balanced",n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        #trial accuracy
        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)
        #recall on ddos for trial # UPDATE TO USE LIBRARY FUNC? (recall_score)
        report = classification_report(y_test, y_pred, output_dict=True)
        recall_attack = report["1"]["recall"]
        recall_scores.append(recall_attack)

    print("\nAccuracy across ", trials, " runs:", acc_scores)
    print("Mean:", np.mean(acc_scores), ", Standard Deviation:", np.std(acc_scores))
    print("\nRecall (attack class = 1) across ",trials," runs: ", recall_scores)
    print("Mean recall: ",np.mean(recall_scores),", Standard Deviation: ",np.std(recall_scores))

# random_uid(): generates a unique random id for output model identification
# - returns 4 characters
def random_uid(k=4):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ23456789"
    return ''.join(random.choices(chars, k=k))

# safe_model_folder(): creates a unique directory using base_name for trained model
def safe_model_folder(base_name, root="models", counter=1):
    while True:
        uid = random_uid()
        folder_name = f"{base_name}_{counter}_{uid}"
        model_dir = os.path.join(root, folder_name)
        if not os.path.exists(model_dir): 
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "model.joblib")
            return folder_name, model_dir, model_path
        counter += 1

# get_user_datasets(): handles user input for dataset selection
# - returns selected dataset names & base file name
def get_user_datasets():
    print("Available datasets:", ", ".join(ATTACK_TYPES))
    while True:
        user_input = input("Enter datasets (comma-separated, or 'ALL'): ").upper().strip()
        if user_input == "ALL": return ATTACK_TYPES, "model_allsets"
        sets = [x.strip() for x in user_input.split(",") if x.strip()]
        #validate
        bad = [k for k in sets if k not in JOBLIB_FILES]
        if bad:
            print(f"[!] ERROR: Invalid dataset keys: {bad}")
            print("Valid options are:", ", ".join(ATTACK_TYPES))
            continue
        if len(sets) == 1: base = f"model_{sets[0]}"
        else:              base = f"model_{len(sets)}sets"
        return sets, base

# append_metrics()
def append_metrics(csv_path, header, row):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file: writer.writerow(header)
        writer.writerow(row)

# MAIN PROGRAM STUFF ===============================================================
# DATASET CODE ----------------------------------
if (DATA_FLAG):
    if (ALL_SETS_FLAG):
        USED_DATASETS = ATTACK_TYPES
        base_name = "model_allsets"
        print("\nTraining model on ALL datasets:")
    else: USED_DATASETS, base_name = get_user_datasets()
    
    # load the specified datasets, collect & append features/labels
    print("Loading datasets now.")
    all_X, all_y = [], []
    for key in USED_DATASETS:
        path = JOBLIB_FILES[key]
        print(f"Loading {path}")
        X_part, y_part = joblib.load(path)
        all_X.append(X_part)
        all_y.append(y_part)
    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    print(", ".join(USED_DATASETS))

    # clean up features
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    X = X.clip(-1e12, 1e12)
    X = X.astype(np.float32)

    # Print info for total datasets being used 
    print("Loaded total dataset:") # should be 111,680 if ALL_SETS_FLAG
    print("Features:", X.shape)
    print("Labels:", y.shape)
    print("Label counts:", y.value_counts())

    name_map = {}
    for col in X.columns:
        if isinstance(col, int) and 0 <= col < len(FEATURE_NAMES): name_map[col] = FEATURE_NAMES[col]
        else: name_map[col] = f"feature_{col}"
    X = X.rename(columns=name_map)

else: 
    # preparing dummy data
    data = genDummyData()
    print(data.head())
    print(data["label"].value_counts()) # num of normal rows, # of ddos rows
    X = data.drop(columns=["label"]) 
    y = data["label"]
    USED_DATASETS = ["DUMMY_DATA"]
    base_name = "model_dummyset"
    print("\nTraining model on dummy dataset")

# MODEL CODE ------------------------------------------
# train/test split
if (RAND_FLAG): X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y)
else: X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=SEED_VAL)
print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)

# Random Forest model
if (RAND_FLAG): rf_model = RandomForestClassifier( n_estimators=200, class_weight="balanced", n_jobs=-1)
else: rf_model = RandomForestClassifier( n_estimators=200, class_weight="balanced", random_state=SEED_VAL, n_jobs=-1)

# train model on train split data
rf_model.fit(X_train, y_train)
print("\nModel trained successfully!")

# create unique folder for this trained model
model_name, model_dir, model_path = safe_model_folder(base_name)
print(f"\nModel run ID: {model_name}")
print("Outputs will be stored in:", model_dir)

# detect/eval test split data
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:,1]

# model reports -------------------------------------------------------------------
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# confusion matrix png
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix ({model_name})")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Benign (0)', 'DDoS (1)'])
plt.yticks(tick_marks, ['Benign (0)', 'DDoS (1)'])
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i,j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i,j] > thresh else "black")
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
plt.show()

# model metrics logging 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

metrics_header = [
    "model_name",
    "datasets_used",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "num_features",
    "timestamp"
]
metrics_row = [
    model_name, 
    ";".join(USED_DATASETS),
    accuracy,
    precision,
    recall,
    f1,
    roc_auc,
    X_train.shape[1],
    pd.Timestamp.now()
]

# metrics records
global_metrics_csv = "reports/model_metrics.csv"
append_metrics(global_metrics_csv, metrics_header, metrics_row)
print(f"\nMetrics appended to {global_metrics_csv}")

per_model_csv = os.path.join(model_dir, "metrics.csv")
append_metrics(per_model_csv, metrics_header, metrics_row)
print(f"Metrics saved to {per_model_csv}")

# model feature importances
feat_importances = rf_model.feature_importances_
feature_importance_dict = dict(
    sorted(
        ((name, float(score)) for name, score in zip(X.columns, feat_importances)),
        key=lambda x: x[1],
        reverse=True
    )
)

print("\n\nFeature Importances:")
for name, score in sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True): print(f"{name}: {score:.4f}")
sorted_items = list(feature_importance_dict.items())
top_items = sorted_items[:20]
feature_names = [name for name, _ in top_items]
top_importances = [score for _, score in top_items]

# feature importances bar chart (top 20)
plt.figure(figsize=(8,6))
plt.barh(range(top_items), top_importances, align="center")
plt.yticks(range(top_items), feature_names)
plt.xlabel("Importance")
plt.title(f"Top Feature Importances ({model_name})")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(model_dir, "feature_importances.png"))
plt.show()

# test rand spread func
#measureRandSpread(10) # i think too many trials caused an error? check that!!

# save trained model to a file
joblib.dump({
    "model": rf_model,
    "datasets_used": USED_DATASETS,
    "feature_names": list(X.columns),
    "feature_importances": feature_importance_dict
}, model_path)
print("\nModel was trained using datasets:", ", ".join(USED_DATASETS))
print(f"\nModel saved to {model_path}")
print("All model artifacts stored in:", model_dir)
print("")

# ------------------------------------------------------
# binary classification (0 = normal, 1 = ddos)
# 
# ********IMPORTANT: Fixed to random, vice versa: rng declaration, df declaration, train/test split, random forest classifier********
# Notes
#   try different n_estimators (& other classifiers) to tune model & improve recall/precision
#   add max_depth to random forest classifier??
#   FOR ACTUAL DATASET: either reduce n_estimators or use smaller data subset. do not melt thy laptop its already funky 
#   for dummy dataset, used short/bursty duration only for ddos (could've also used extremely long/stuck), think about adding both in? 
#   **Look into standard scaling??? 
#
# TODO:
#       - add more metrics stuff? reports, etc
#       - add in logistic regression and neural network
# -------------------------------------------------------
#   TUTORIAL BASE USED: https://www.labellerr.com/blog/ddos-attack-detection/#building-a-ddos-detection-model