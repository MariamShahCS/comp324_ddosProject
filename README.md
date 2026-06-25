# DDoS Attack Detection Framework

## Overview

This project implements a modular machine learning pipeline for detecting Distributed Denial-of-Service (DDoS) attacks using supervised learning techniques. The project was developed as part of a university Computer Science course and focuses on building reusable tooling for preprocessing network traffic datasets, training multiple machine learning models, and evaluating their performance using common classification metrics.

Rather than focusing on a single classifier, the project was designed to support experimentation across multiple DDoS datasets and machine learning approaches.

---

## Features

* Modular preprocessing pipeline for network traffic datasets
* Support for multiple DDoS attack datasets
* Configurable dataset selection
* Balanced training dataset generation
* Random Forest, Logistic Regression, and Neural Network classifiers
* Automatic model serialization using Joblib
* Experiment logging
* Classification reports
* Confusion matrix generation
* ROC curve generation
* Feature importance visualization (supported models)

---

## Technologies

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Joblib

---

## Supported Models

* Random Forest
* Logistic Regression
* Multi-Layer Perceptron (Neural Network)

---

## Supported Datasets

The framework currently supports the following DDoS attack datasets:

* LDAP
* MSSQL
* NETBIOS
* SYN
* UDP
* UDPLAG

Datasets are preprocessed individually and can be trained independently or combined for experimentation.

---

## Project Structure

prep_training_data.py

* Loads raw parquet datasets
* Cleans and converts features
* Balances benign and attack traffic
* Aligns train/test features
* Exports preprocessed datasets as Joblib files

supervised_learning_model.py

* Loads preprocessed datasets
* Trains the selected machine learning model
* Evaluates classifier performance
* Generates confusion matrices
* Generates ROC curves
* Saves trained models and experiment metrics

model_tester.py

* Work in progress for comparing multiple trained models using a shared evaluation pipeline.

---

## Evaluation

Each trained model is evaluated using several standard machine learning metrics:

* Accuracy
* Precision
* Recall
* F1-score
* ROC AUC
* Confusion Matrix

For supported models, feature importance visualizations are also generated.

---

## Results

Example outputs include:

* Classification reports
* Confusion matrices
* ROC curve visualizations
* Feature importance charts
* Experiment logs
* Saved trained models

---

## Future Improvements

Planned improvements include:

* Complete the multi-model comparison framework
* Hyperparameter tuning
* Cross-validation
* Additional DDoS datasets
* Expanded experiment reporting
* Improved documentation and visualization

---

## Disclaimer

This project was developed for educational purposes as part of a university machine learning project focused on network intrusion detection and supervised learning techniques.
