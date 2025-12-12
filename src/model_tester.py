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

from sklearn.metrics import roc_curve, auc
rf_proba = rf_model.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test, rf_proba[:, 1]) 

