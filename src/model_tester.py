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



#  1. Find all the model files:
   # Look inside the "models" folder.
   # For each subfolder, look for a file named "model.joblib".
   # Make a list of these file paths.

#  2. Load the test data:
    # Use joblib.load("some_test_data.joblib") to get:
    #   X_test = input features
    #   y_test = true labels (0 or 1)

# 3. For each model file in the list:

    #  a) Load the model info:
    #  model_bundle = joblib.load(model_file)
    #  model = model_bundle["model"]
    #  scaler = model_bundle["scaler"]  (might be None)
    #  feature_names = model_bundle["feature_names"]

  # b) Make sure X_test has the same columns in the same order
    #  that the model was trained on.
    #  (You can reorder X_test columns using feature_names.)

  # c) If there is a scaler:
      # Use scaler to transform X_test before prediction.

  # d) Ask the model to predict:
      # y_pred = model.predict(X_test)

  # e) Compare y_pred to y_test:
      # Count how many are correct (this is accuracy).
      # Optionally compute other scores if you want (precision, recall, etc).

  # f) Print a short line like:
      # - "Model XYZ: accuracy = 0.95, recall = 0.92"

# 4. At the end, you will have several lines, one per model,
     # so you can see which one looks best.
