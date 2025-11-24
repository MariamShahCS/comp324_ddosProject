import os, sys, math
import numpy as np
import pandas as pd
import joblib

from collections import Counter
from .Connection import load_pickle, get_match_rows
#
#DATA_FILES = ["DrDos_DNS.gz", "DrDos_LDAP.gz", "DrDos_MSSQL.gz", "DrDos_NetBIOS.gz", "DrDos_NTP.gz", "DrDos_SNMP.gz", "DrDos_SSDP.gz", "DrDos_UDP.gz", "Syn.gz", "TFTP.gz", "UDPLag.gz"]
DATA_FILES = "data/DrDoS_UDP_74len.gz"
#LABEL_COL = 87
FLOWID_COL = 1
SEED_VAL = 42
BALANCE_FLAG = True
ATTACK_MULTIPLIER = 1

rng = np.random.default_rng(SEED_VAL)
overall_counts = Counter()
per_file = []
benign_all = []
attacks_all = []
n_attack = 0
n_benign = 0

'''for fname in (DATA_FILES):
    if not os.path.exists(fname): 
        print(f"File {fname} not found, skipping.")
        continue
    try: 
        print(f"[load] {fname} ...")
        rows = load_pickle(fname)   # list of lists
        n = len(rows)
        labels = [r[LABEL_COL] for r in rows]
        c = Counter(labels)
        overall_counts.update(c)
        per_file.append((fname, n, dict(c)))

        benign_rows = get_match_rows(rows, LABEL_COL, 0)
        benign_all.extend(benign_rows)

        attacks_rows = [r for r in rows if r[LABEL_COL] != 0]
        attacks_all.extend(attacks_rows)

        print(f"  rows: {n:,}, label counts: {dict(c)} | benign in file: {len(benign_rows):,}")

    except Exception as e: 
        print(f"[error] {fname}: {e}")'''

print(f"[load] {DATA_FILES} ...")
rows = load_pickle(DATA_FILES)
n = len(rows)

if not rows:
    print("[!] WARNING: No rows loaded from file, ABORT.")
    sys.exit(1)

num_cols = len(rows[0])
print(f"Loaded {n:,} rows with {num_cols} columns each")
LABEL_COL = num_cols - 1

filtered_rows = [r for r in rows if len(r) > LABEL_COL]
if len(filtered_rows) < len(rows):
    print(f"[!] WARNING: Dropped {len(rows) - len(filtered_rows)} rows with too few columns.")
rows = filtered_rows

labels = [r[LABEL_COL] for r in rows]
c = Counter(labels)
overall_counts.update(c)
per_file.append((DATA_FILES, n, dict(c)))

benign_rows = get_match_rows(rows, LABEL_COL, 0)
benign_all.extend(benign_rows)

attacks_rows = [r for r in rows if r[LABEL_COL] != 0]
attacks_all.extend(attacks_rows)

print(f"    rows: {n:,}, label counts: {dict(c)} | benign in file: {len(benign_rows):,}")


print("\n SUMMARY (per file):")
for fname, n, c in per_file: print(f"{fname:18s} -> rows={n:,} labels={c}")

print("\nOVERALL SUMMARY:")
print (dict(overall_counts))

print(f"\nTotal benign: {len(benign_all):,}")
print(f"\nTotal Attacks: {len(attacks_all):,}")

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

combined = benign_all[:n_benign] + attack_sample
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

out_path = "data/training_network_data.joblib"
joblib.dump((X,y), out_path)
print(f"\nTraining data saved to {out_path}!")