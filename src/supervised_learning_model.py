import pandas as pd
import numpy as np
import joblib
import os, sys, math

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc

from collections import Counter
from Connection import load_pickle, get_match_rows

# GLOBAL VARIABLES =========================================
SEED_VAL = 42 # for random seeding/testing reproducability
RAND_FLAG = True   # if random set True, else if fixed set False
DATA_FLAG = True   # if using dummyData set False, if using actual data set True
DATA_FILES = ["DrDos_DNS.gz", "DrDos_LDAP.gz", "DrDos_MSSQL.gz", "DrDos_NetBIOS.gz", "DrDos_NTP.gz", "DrDos_SNMP.gz", "DrDos_SSDP.gz", "DrDos_UDP.gz", "Syn.gz", "TFTP.gz", "UDPLag.gz"]

# FUNCTION DEFINITIONS ==================================
# genDummyData(): generate dummy dataset with super simple indicators
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

# measureRandSpread(): measure random spread accuracy, only rand trials
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

# UDP LAG test first only
# MAIN PROGRAM STUFF ===============================================================
# DATASET CODE ----------------------------------
# for real dataset: df = pd.read_csv("set.csv")
# data preprocessing? 
# df.columns = df.columns.str.strip()
# df.loc[:,'Label'].unique()
# df=df.dropna() 
# (df.dtypes=='object')
# df['Label'] = df['Label'].map({'normal':0, 'ddos':1})
# should i use matplotlib?? (can use to check/visualize distribution of classes)

if (DATA_FLAG):
    print("Loading DrDoS_UDP.gz... please wait...")
    X, y = joblib.load("training_network_data.joblib")
    print("Loaded features:", X.shape)
    print("Loaded labels:,", y.shape)
    print("Label counts:", y.value_counts())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    X = X.clip(-1e12, 1e12)

else: 
    data = genDummyData()
    print(data.head())
    print(data["label"].value_counts()) # num of normal rows, # of ddos rows

# MODEL CODE ------------------------------------------
# train/test split
if (not DATA_FLAG):
    X = data.drop(columns=["label"]) 
    y = data["label"]

if (RAND_FLAG): X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y)
else: X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=SEED_VAL)
print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)

# Random Forest model
if (RAND_FLAG): rf_model = RandomForestClassifier( n_estimators=200, class_weight="balanced", n_jobs=-1)
else: rf_model = RandomForestClassifier( n_estimators=200, class_weight="balanced", random_state=SEED_VAL, n_jobs=-1)

# train model on train split data
rf_model.fit(X_train, y_train)
print("\nModel trained successfully!")

# detect/eval test split data
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:,1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# feature importance
feat_importances = rf_model.feature_importances_ 
for name, score in zip(X.columns, feat_importances): print(f"{name}:{score:.4f}")

#og = [578324]
#new = [0, ] # binary based on threshold (normal vs ddos)


# test rand spread func
#measureRandSpread(10) # i think too many trials caused an error? check that!!

# save trained model to a file
joblib.dump(rf_model, "supervised_learning_ddos_detector.joblib")
print("\nModel saved to supervised_learning_ddos_detector.joblib")

# ------------------------------------------------------
# ^^^ wrote a dummy dataset generator using numpy, created a data frame using pandas
# ^^^   distributed dummy network traffic from dataset for test/train split (features=X, labels=y)
#       random forest model of 200 trees, balanced distribution in case not many ddos samples (if normal/ddos is imbalanced), currently uses all cores
#       binary classification (0 = normal, 1 = ddos)
# 
# ********IMPORTANT: Fixed to random, vice versa: rng declaration, df declaration, train/test split, random forest classifier********
# Notes
#   replace dummy dataset
#   try different n_estimators (& other classifiers) to tune model & improve recall/precision
#   work on visualization?? (do something about trial prints?)
#   add max_depth to random forest classifier??
#   probably might need to write preprocessing code??? check tutorial once have real dataset?
#   will possibly need to use strip(), dropna(), map(), etc??? diff df, everything else (model logic) should mostly be same tho.
#       FOR ACTUAL DATASET: either reduce n_estimators or use smaller data subset. do not melt thy laptop its already funky 
#   for dummy dataset, used short/bursty duration only for ddos (could've also used extremely long/stuck), think about adding both in? 
#   **Look into standard scaling??? 

# write stuff to save results/metrics, and to save experiment stability (GOOD FOR REPORTS/WRITE UPS/PRESENTATION)

# should i also use logistic regression/neural network??

# joblib stuff:
# to eval w/o retraining in new file:
#   import joblib & pandas
# -> rf_model = joblib.load("ddos_detector.joblib")
# -> load new data to eval: data = pd.DataFrame({...})
# -> evals = rf_model.predict(data), print(evals) # EX output: [0 1] 
# -------------------------------------------------------
#   TUTORIAL BASE USED: https://www.labellerr.com/blog/ddos-attack-detection/#building-a-ddos-detection-model