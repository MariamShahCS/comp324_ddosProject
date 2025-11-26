import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, os, random, csv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from .features import FEATURE_NAMES

# GLOBAL VARIABLES ======================================================================================================
SEED_VAL = 42           # for random seeding/testing reproducability, you hitchhiker

# FLags -----------------------------------------------------------------------------------------------------------------
ALL_SETS_FLAG = False   # if using all (real) datasets set True, elif want user input to select datasets set False
DATA_FLAG = False        # if using dummyData set False, elif using real data set True
MODEL_FLAG = False       # if using default model set False, elif want user to select model set True
RAND_FLAG = True        # if random set True, elif fixed set False

# Lists -----------------------------------------------------------------------------------------------------------------
ATTACK_TYPES = ["DNS","LDAP","MSSQL","NETBIOS","NTP","SNMP","SSDP","SYN","TFTP","UDP","UDPLAG"]
MODEL_TYPES_LIST = ["RF", "LR", "NN"]

# Dictionaries
JOBLIB_FILES = {
    key:f"data/joblibs/training_{key}_data.joblib"
    for key in ATTACK_TYPES
}
MODEL_NAMES = {
    "RF":"Random Forest",
    "LR":"Logistic Regression",
    "NN":"Neural Network"
}

# Other
MODEL_TYPE = MODEL_TYPES_LIST[0] # set default model to random forest

# PATH CHECKS ============================================================================================================
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# FUNCTION DEFINITIONS ===================================================================================================

# dummy functions---------------------------------------------------------------------------------------------------------
# genDummyData(): generate dummy dataset with super simple attack indicators
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
    
# Utility functions--------------------------------------------------------------------------------------------------------
# append_metrics(): appends model metrics to a csv file
def append_metrics(csv_path, header, row):
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file: writer.writerow(header)
        writer.writerow(row)

# random_uid(): generates a unique random id for output model identification
# - returns 4 characters
def random_uid(k=4):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
    return ''.join(random.choices(chars, k=k))

# safe_model_folder(): creates a unique directory using base_name for trained model
# - returns a unique folder name for the model, creates a directory to hold model artifacts, and a path for the joblib file
def safe_model_folder(base_name, root="models"):
    while True:
        uid = random_uid()
        folder_name = f"{base_name}_{uid}"
        model_dir = os.path.join(root, folder_name)

        if not os.path.exists(model_dir): 
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "model.joblib")
            return folder_name, model_dir, model_path

# Plotting functions--------------------------------------------------------------------------------------------------------
# finalize_plot(): finalizes and saves matplotlib plots
def finalize_plot(model_dir, filename, message_prefix=None):
    path = os.path.join(model_dir, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    if message_prefix: print(f"{message_prefix} saved to {path}")

# Modeling functions--------------------------------------------------------------------------------------------------------
# choose_model_type(): validates user input to select model type
def choose_model_type():
    print("\nAvailable model types:")
    for m in MODEL_TYPES_LIST: print(f"  {m} ({MODEL_NAMES[m]})")
    while True:
        choice = input("Select model type: ").upper().strip()
        if choice in MODEL_TYPES_LIST: return choice
        valid_text = ", ".join(f"{m} ({MODEL_NAMES[m]})" for m in MODEL_TYPES_LIST)
        print(f"[!] ERROR: Invalid choice '{choice}'. Valid options: {valid_text}")

# get_feature_importance_dict(): extracts & sorts feature importances for supported models
# - returns a dict of model feature importances
# - returns None if model doesn't provide importances
def get_feature_importance_dict(model, X):
    # check if model has feature_importances_ or coef_ attribute
    if hasattr(model, "feature_importances_"): importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if coef.ndim > 1: coef = np.mean(np.abs(coef), axis=0)
        else: coef = np.abs(coef)
        importances = coef
    else: 
        print("\n[!] WARNING: This model type does not expose feature importances; skipping.")
        return None
    feature_importance_dict = dict(
        sorted(
            ((name, float(score)) for name, score in zip(X.columns, importances)),
            key=lambda x: x[1],
            reverse=True
        )
    )
    return feature_importance_dict

# MAIN PROGRAM START =======================================================================================================
# DATASET CODE -------------------------------------------------------------------------------------------------------------
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
    print("Loaded total dataset:")
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

# MODEL CODE -----------------------------------------------------------------------------------------------------------------
# train/test split
if (RAND_FLAG): X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y)
else: X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=SEED_VAL)
print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)

# select model
if MODEL_FLAG: MODEL_TYPE = choose_model_type()

# standard scaling (for LR and NN)
scaler = None
if MODEL_TYPE in ["LR", "NN"]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# build model
if MODEL_TYPE == "RF":
    # Random Forest model
    if (RAND_FLAG): model = RandomForestClassifier( n_estimators=200, class_weight="balanced", n_jobs=-1)
    else: model = RandomForestClassifier( n_estimators=200, class_weight="balanced", random_state=SEED_VAL, n_jobs=-1)
elif MODEL_TYPE == "LR":
    # Logistic Regression model
    if (RAND_FLAG): model = LogisticRegression( max_iter=1000, class_weight="balanced")
    else: model = LogisticRegression( max_iter=1000, class_weight="balanced", random_state=SEED_VAL)
elif MODEL_TYPE == "NN":
    # Neural Network Model
    if (RAND_FLAG): model = MLPClassifier( hidden_layer_sizes=(128,64,32), activation='relu', solver='adam', max_iter=300, learning_rate='adaptive')
    else: model = MLPClassifier(hidden_layer_sizes=(128,64,32), activation='relu', solver='adam', max_iter=300, learning_rate='adaptive', random_state=SEED_VAL)

# train model on train split data
model.fit(X_train, y_train)
print(f"\n{MODEL_NAMES[MODEL_TYPE]} Model trained successfully!")

# create unique folder for this trained model
base_name = f"{base_name}_{MODEL_TYPE}"
model_name, model_dir, model_path = safe_model_folder(base_name)
print(f"\nModel run ID: {model_name}")
print("Outputs will be stored in:", model_dir)

# detect/eval test split data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# MODEL REPORTS ===================================================================================================================
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# save confusion matrix 
cm_df = pd.DataFrame(
    cm,
    index=['True_0', 'True_1'],
    columns=['Pred_0', 'Pred_1']
)
cm_csv_path = os.path.join(model_dir, "confusion_matrix.csv")
cm_df.to_csv(cm_csv_path)
print(f"Confusion matrix saved to {cm_csv_path}")

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
finalize_plot(model_dir, "confusion_matrix.png")

# calc model metrics 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# save ROC curve
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve ({model_name})")
plt.legend(loc="lower right")
finalize_plot(model_dir, "roc_curve.png", message_prefix="ROC curve")

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
feature_importance_dict = get_feature_importance_dict(model, X)

if feature_importance_dict is not None:
    print("\n\nFeature Importances:")
    for name, score in feature_importance_dict.items(): print(f"{name}: {score:.4f}")
    top_items = list(feature_importance_dict.items())[:20]
    feature_names = [name for name, _ in top_items]
    top_importances = [score for _, score in top_items]

    # feature importances bar chart (top 20)
    plt.figure(figsize=(8,6))
    plt.barh(range(len(top_items)), top_importances, align="center")
    plt.yticks(range(len(top_items)), feature_names)
    plt.xlabel("Importance")
    plt.title(f"Top Feature Importances ({model_name})")
    plt.gca().invert_yaxis()
    finalize_plot(model_dir, "feature_importances.png")
else: feature_importance_dict = {}

# SAVE TRAINED MODEL TO JOBLIB FILE ==========================================================================================================
joblib.dump({
    "model": model,
    "scaler": scaler if MODEL_TYPE in ["LR", "NN"] else None,
    "model_type": MODEL_TYPE,
    "datasets_used": USED_DATASETS,
    "feature_names": list(X.columns),
    "feature_importances": feature_importance_dict
}, model_path)
print("\nModel was trained using datasets:", ", ".join(USED_DATASETS))
print(f"\nModel saved to {model_path}")
print("All model artifacts stored in:", model_dir)
print("")

# =============================================================================================================================================
# binary classification (0 = normal, 1 = ddos)
# 
# ********IMPORTANT: Fixed to random, vice versa: rng declaration, df declaration, train/test split, random forest classifier********
# Notes
#   try different n_estimators (& other classifiers) to tune model & improve recall/precision
#   add max_depth to random forest classifier??
#   for dummy dataset, used short/bursty duration only for ddos (could've also used extremely long/stuck), think about adding both in? 
#
# TODO:
#       - add more metrics stuff? reports, etc
# -------------------------------------------------------
#   TUTORIAL BASE USED: https://www.labellerr.com/blog/ddos-attack-detection/#building-a-ddos-detection-model