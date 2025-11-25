import os, sys, joblib, time, csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from .Connection import load_pickle

# create a dictionary tracking which dataset is being used
ATTACK_TYPES = ["DNS","LDAP","MSSQL","NETBIOS","NTP","SNMP","SSDP","SYN","TFTP","UDP","UDPLAG"]
DATA_FILES = {
    key: {
        "raw":f"data/raw/DrDoS_{key}_74len.gz",
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
DATA_FILE = config["raw"]
LABEL_COL = None
FLOWID_COL = 1
SEED_VAL = 42
BALANCE_FLAG = True
ATTACK_MULTIPLIER = 1

rng = np.random.default_rng(SEED_VAL)
benign_all = []
attacks_all = []

# load and start prepping the data
start = time.time()

# load data file in
print(f"[load] {DATA_FILE} ...")
if not os.path.exists(DATA_FILE):
    print(f"[!] ERROR: file {DATA_FILE} not found, ABORT.")
    sys.exit(1)
try:
    rows = load_pickle(DATA_FILE)
except Exception as e:
    print(f"[!] ERROR: failed to load {DATA_FILE}: {e}")
    sys.exit(1)

n = len(rows)
if not rows:
    print("[!] WARNING: No rows loaded from file, ABORT.")
    sys.exit(1)

num_cols = len(rows[0])
print(f"Loaded {n:,} rows with {num_cols} columns each")
LABEL_COL = num_cols - 1

# ensure all rows are usable
filtered_rows = [r for r in tqdm(rows, desc="Filtering invalid rows") if len(r) > LABEL_COL]
dropped = len(rows) - len(filtered_rows)
if dropped > 0:
    print(f"[!] WARNING: Dropped {dropped:,} bad rows.")
rows = filtered_rows

# sort the rows
benign_rows = [r for r in tqdm(rows, desc="Extracting benign traffic") if r[LABEL_COL] == 0]
benign_all.extend(benign_rows)

attacks_rows = [r for r in tqdm(rows, desc="Extracting attack traffic") if r[LABEL_COL] != 0]
attacks_all.extend(attacks_rows)

# print file info
labels = [r[LABEL_COL] for r in rows]
c = Counter(labels)
print("\n--- Dataset Summary ---")
print(f"File: {DATA_FILE}")
print(f"Total rows loaded: {n:,}")
print(f"Label distribution: {dict(c)}")
print(f"Benign samples: {len(benign_rows):,}")
print(f"Attack samples: {len(attacks_all):,}")

if len(benign_all) == 0:
    print("\n[!] WARNING: NO NORMAL TRAFFIC FOUND")
    sys.exit(1)
elif len(attacks_all) == 0:
    print("\n[!] WARNING: NO ATTACK TRAFFIC FOUND")
    sys.exit(1)
    
# build a balanced and mixed data subset
if BALANCE_FLAG:
    n_benign = len(benign_all)
    n_attack = min(len(attacks_all), n_benign * ATTACK_MULTIPLIER)
else:
    n_benign = min(len(benign_all), 50_000)
    n_attack = min(len(attacks_all), n_benign * ATTACK_MULTIPLIER)

print(f"\nBuilding dataset with benign={n_benign:,}, attack={n_attack:,} (multiplier={ATTACK_MULTIPLIER})")

attack_idx = rng.choice(len(attacks_all), size=n_attack, replace=False)
attack_sample = [attacks_all[i] for i in attack_idx]

combined = []
for r in tqdm(benign_all[:n_benign], desc="Adding benign samples"):
    combined.append(r)
for r in tqdm(attack_sample, desc="Adding attack samples"):
    combined.append(r)

df = pd.DataFrame(combined)

y = (df.iloc[:, LABEL_COL] != 0).astype(int) # 0 = benign, 1 = attack

drop_cols = []
if FLOWID_COL < df.shape[1]:
    drop_cols.append(df.columns[FLOWID_COL])
drop_cols.append(df.columns[LABEL_COL])

X = df.drop(columns=drop_cols, errors="ignore")
X = X.apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1, how="all").fillna(0)

print(f"Features shape: {X.shape} | Labels: {y.value_counts().to_dict()}")

print("\n")
print(X.head())
print(X.columns)
# output joblib file
out_path = config["joblib"]
joblib.dump((X,y), out_path)

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
                 n,
                 len(benign_all),
                 len(attacks_all),
                 X.shape[1],
                 len(X),
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