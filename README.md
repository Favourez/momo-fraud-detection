Momo Fraud Detection Dataset – AI & Data Science Pipeline
Project Overview

This project implements an end-to-end data science and AI pipeline in Python to detect fraudulent mobile money transactions using the Momo Fraud Detection dataset. The system simulates a real-world workflow by integrating:

Data acquisition & preprocessing

Custom risk scoring using rules

Handling class imbalance with SMOTE

Machine Learning model training and evaluation (Random Forest)

Visualization of results and feature importance

The goal is to detect rare fraudulent transactions (~0.1–0.2% of all transactions) accurately, which is a key challenge in financial fraud detection.

Problem Statement

Mobile money fraud is a major issue, especially with increasing digital transactions. Detecting fraudulent transactions in real time is critical for financial institutions and mobile operators.

Key challenges:

Extremely imbalanced dataset

High variability in transaction types and amounts

Need for combining rule-based risk features with machine learning

This project demonstrates how to build a reliable AI-based fraud detection pipeline using Python.

Dataset

Name: momo-fraud-detection-dataset.csv
Source: Kaggle – PaySim Synthetic Mobile Money Dataset

Columns include:

Column	Description
step	Time step of transaction
type	Transaction type (PAYMENT, TRANSFER, CASH_OUT, etc.)
amount	Transaction amount
nameOrig	Sender account ID
oldbalanceOrg	Sender balance before transaction
newbalanceOrig	Sender balance after transaction
nameDest	Recipient account ID
oldbalanceDest	Recipient balance before transaction
newbalanceDest	Recipient balance after transaction
isFraud	Target variable (1 = fraud, 0 = not fraud)
isFlaggedFraud	Flagged fraud (manual)

Observation:
Fraud transactions are extremely rare (~0.1%), making this a highly imbalanced classification problem.

Project Structure
momo-fraud-detection/
│
├─ data/
│   └─ momo-fraud-detection-dataset.csv
│
├─ src/
│   ├─ risk_algorithm.py         # Custom FraudRiskScorer class
│   ├─ model.py                  # (Optional) ML model class if used
│   ├─ preprocessing.py          # (Optional) preprocessing functions
│   ├─ data_loader.py            # (Optional) dataset loading functions
│   └─ visualization.py          # (Optional) plotting functions
│
├─ notebook.ipynb                # Main notebook (EDA, SMOTE, ML, evaluation)
├─ README.md                     # Project overview and instructions
└─ requirements.txt              # Python dependencies

Installation & Setup

Clone the repository:

git clone <your-repo-url>
cd momo-fraud-detection


Install dependencies:

pip install -r requirements.txt


Requirements example (requirements.txt):

pandas
numpy
scikit-learn
imbalanced-learn
matplotlib


Launch the notebook:

jupyter notebook notebook.ipynb

Workflow

Load dataset

Exploratory Data Analysis (EDA) – check data types, distributions, and imbalanced classes

Apply custom risk algorithm (FraudRiskScorer) to create risk_score feature

Encode categorical features (transaction type)

Split dataset into train/test

Balance training data with SMOTE

Train Random Forest classifier on resampled data

Evaluate model using precision, recall, F1-score, and confusion matrix

Visualize feature importance

Results & Observations

Class imbalance before SMOTE:

0: 5,083,526
1: 6,570


After SMOTE (training data):

0: 5,083,526
1: 5,083,526


Random Forest evaluation (example):

Precision: high for fraud class after SMOTE

Recall: improved for detecting frauds

Feature importance: risk_score, amount, and type contribute most

SMOTE ensures the model can learn patterns from rare fraud transactions, and the risk score feature helps boost performance.

Conclusion

Mobile money fraud detection is challenging due to highly imbalanced data.

Combining rule-based features with machine learning models improves detection.

SMOTE is critical to balance the dataset for training.

Random Forest provides a good baseline for fraud detection.

Next Steps / Improvements

Experiment with XGBoost or LightGBM for faster training and higher accuracy

Tune Random Forest hyperparameters (n_estimators, max_depth)

Explore real-time fraud scoring for live transaction streams

Credits

Dataset: Kaggle – PaySim1

Python libraries: Pandas, NumPy, scikit-learn, imbalanced-learn, Matplotlib