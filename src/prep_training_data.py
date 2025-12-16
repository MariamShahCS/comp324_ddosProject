import os, sys, joblib, time, csv
import pandas as pd
from collections import Counter

# clean and build dataframe
def build_Xy(df: pd.DataFrame, label_col: str, flowid_col):
    y = (df[label_col].astype(str).str.strip().str.lower() != "benign").astype(int)

    drop_cols = [label_col]
    if flowid_col is not None and isinstance(flowid_col, int) and flowid_col < df.shape[1]:
        drop_cols.append(df.columns[flowid_col])
    elif isinstance(flowid_col, str) and flowid_col in df.columns:
        drop_cols.append(flowid_col)

    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all").fillna(0)
    return X, y


# create a dictionary tracking which dataset is being used
ATTACK_TYPES = ["LDAP","MSSQL","NETBIOS","SYN","UDP","UDPLAG"]
DATA_FILES = {
    key: {
        "train":f"data/raw/{key}_training.parquet",
        "test":f"data/raw/{key}_testing.parquet",
        "joblib":f"data/joblibs/training_{key}_data.joblib"
    }
    for key in ATTACK_TYPES
}

# allow user input to choose which dataset to prep
print("Available datasets:", ", ".join(DATA_FILES.keys()))
FILE_KEY = input("Enter which dataset you'd like: ").upper().strip()
while FILE_KEY not in DATA_FILES:
    print(f"[!] ERROR: dataset '{FILE_KEY}' not listed.")
    print(" Please choose from",", ".join(DATA_FILES.keys()))
    FILE_KEY = input("Enter which dataset you'd like: ").upper().strip()

# create vars to track stuff
config = DATA_FILES[FILE_KEY]
TRAIN_FILE = config["train"]
TEST_FILE = config["test"]
LABEL_COL = "Label"
FLOWID_COL = None
SEED_VAL = 42
BALANCE_FLAG = True
ATTACK_MULTIPLIER = 1

# load and start prepping the data
start = time.time()

# load parquet train/test
print(f"[load train] {TRAIN_FILE}")
if not os.path.exists(TRAIN_FILE):
    print(f"[!] ERROR: file {TRAIN_FILE} not found, ABORT.")
    sys.exit(1)

print(f"[load test] {TEST_FILE}")
if not os.path.exists(TEST_FILE):
    print(f"[!] ERROR: file {TEST_FILE} not found, ABORT.")
    sys.exit(1)

df_train = pd.read_parquet(TRAIN_FILE)
df_test = pd.read_parquet(TEST_FILE)

if LABEL_COL not in df_train.columns or LABEL_COL not in df_test.columns:
    print(f"[!] ERROR: '{LABEL_COL}' column not found in train/test, ABORT.")
    sys.exit(1)

print(f"Loaded {len(df_test):,} (TEST) rows with {df_test.shape[1]} columns each")
print(f"Loaded {len(df_train):,} (TRAIN) rows with {df_train.shape[1]} columns each")

# balance train
label = df_train[LABEL_COL].astype(str).str.strip().str.lower()
benign_df = df_train[label == "benign"]   
attacks_df = df_train[label != "benign"] 

if BALANCE_FLAG:
    
    c = Counter(label)
    print("\n--- Training Data Summary ---")
    print(f"File: {TRAIN_FILE}")
    print(f"Total rows loaded: {len(df_train):,}")
    print(f"Label distribution: {dict(c)}")
    print(f"Benign samples: {len(benign_df):,}")
    print(f"Attack samples: {len(attacks_df):,}")

    n_benign = len(benign_df)
    n_attacks = len(attacks_df)

    target_benign = min(n_benign, n_attacks)
    target_attacks = min(n_attacks, target_benign*ATTACK_MULTIPLIER)

    print(f"\nBuilding dataset with benign={target_benign:,}, attack={target_attacks:,} (multiplier={ATTACK_MULTIPLIER})")

    benign_sample = benign_df.sample(n=target_benign, replace=False, random_state=SEED_VAL)
    attacks_sample = attacks_df.sample(n=target_attacks, replace=False, random_state=SEED_VAL)


    df = pd.concat([benign_sample, attacks_sample], ignore_index=True)
    df = df.sample(frac=1.0, random_state=SEED_VAL).reset_index(drop=True)
else:
    df = df_train
    print(f"\nBuilding dataset with benign={len(benign_df):,}, attack={len(attacks_df):,}")

X_train, y_train = build_Xy(df, LABEL_COL, FLOWID_COL)
X_test, y_test = build_Xy(df_test, LABEL_COL, FLOWID_COL)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# output joblib file
# joblib contains: X_train, y_train, X_test, y_test
out_path = config["joblib"]
os.makedirs(os.path.dirname(out_path), exist_ok=True)
joblib.dump(
    {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test},
    out_path
)

# duration calc
print(f"\nTraining data saved to {out_path}!")
duration = time.time() - start
if duration > 3600:
    hours, rem = divmod(duration, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Completed in {int(hours)}h {int(mins)}m {secs:.1f}s")
else:
    mins, secs = divmod(duration, 60)
    print(f"Completed in {int(mins)}m {secs:.1f}s")

# tracker/records csv file
timing_record = [FILE_KEY,
                 len(df_train),
                 int((y_train == 0).sum()),
                 int((y_train == 1).sum()),
                 X_train.shape[1],
                 len(X_train),
                 round(duration, 2)]

csv_path = "reports/preprocessing_times.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["dataset", 
                         "rows_loaded", 
                         "benign_count",
                         "attack_count",
                         "features",
                         "total_samples",
                         "seconds"])
    writer.writerow(timing_record)
print(f"datafile info appended to {csv_path}")