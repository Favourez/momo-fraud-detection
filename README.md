# **Momo Fraud Detection – AI & Data Science Pipeline**
## **Project Overview**

This project implements an end-to-end Data Science and Artificial Intelligence pipeline in Python to detect fraudulent mobile money (MoMo) transactions using the Momo Fraud Detection dataset.
It simulates a real-world fraud detection workflow by integrating:

Data acquisition and preprocessing

Custom rule-based fraud risk scoring

Handling class imbalance using SMOTE

Machine Learning model training and evaluation (Random Forest)

Visualization of results and feature importance

The primary objective is to accurately identify rare fraudulent transactions (~0.1–0.2%), which represents a major challenge in financial fraud detection systems.

---

## Problem Statement

Mobile money fraud is a growing problem due to the rapid increase in digital financial transactions. Fraudulent activities can lead to significant financial losses, reduced customer trust, and regulatory challenges for mobile operators and financial institutions.

Key challenges addressed in this project include:

Extremely imbalanced data, where fraud cases are very rare

High variability in transaction types and transaction amounts

The need to combine rule-based domain knowledge with machine learning models

This project demonstrates how an AI-driven fraud detection system can be designed using Python to address these challenges effectively.

---

## Dataset

**Name: momo-fraud-detection-dataset.csv**

**Source: Kaggle – PaySim Synthetic Mobile Money Dataset**

[PaySim Mobile Money Fraud Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1/data)


---

## Dataset Features
Column Name	Description
step	Time step of the transaction
type	Transaction type (PAYMENT, TRANSFER, CASH_OUT, etc.)
amount	Transaction amount
nameOrig	Sender account ID
oldbalanceOrg	Sender balance before transaction
newbalanceOrig	Sender balance after transaction
nameDest	Recipient account ID
oldbalanceDest	Recipient balance before transaction
newbalanceDest	Recipient balance after transaction
isFraud	Target variable (1 = Fraud, 0 = Not Fraud)
isFlaggedFraud	Manually flagged fraud

---

## Observation:
Fraudulent transactions represent ~0.1% of the dataset, making this a highly imbalanced classification problem.

---

## Project Structure
momo-fraud-detection/
│
├── data/
│   └── momo-fraud-detection-dataset.csv   # Dataset (not tracked in GitHub)
│
├── src/
│   ├── risk_algorithm.py      # Custom FraudRiskScorer
│   ├── model.py               # ML model logic
│   ├── preprocessing.py       # Data preprocessing functions
│   ├── data_loader.py         # Dataset loading utilities
│   └── visualization.py       # Plotting and visualization utilities
│
├── notebooks/
│   └── notebook.ipynb         # EDA, SMOTE, ML training & evaluation
│
├── tests/
│   └── test_algorithms.py     # Unit tests
│
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies

---

## Installation & Setup
1. Clone the Repository
git clone https://github.com/Favourez/momo-fraud-detection.git
cd momo-fraud-detection

2. Install Dependencies
pip install -r requirements.txt

3. Download Dataset

Download the dataset from Kaggle and place it in:

data/momo-fraud-detection-dataset.csv

4. Run the Notebook
jupyter notebook notebooks/notebook.ipynb

---

## Workflow

- Load the dataset

- Perform Exploratory Data Analysis (EDA)

- Analyze class imbalance

- Apply custom FraudRiskScorer to generate a risk_score feature

- Encode categorical features

- Split data into training and testing sets

- Balance the training data using SMOTE

- Train a Random Forest classifier

- Evaluate model performance

- Visualize confusion matrix and feature importance

---

## Results & Observations
### Class Distribution (Before SMOTE)

Non-Fraud (0): 5,083,526

Fraud (1): 6,570

### After SMOTE (Training Data)

Non-Fraud (0): 5,083,526

Fraud (1): 5,083,526

Model Performance (Random Forest)

High recall for fraud detection, ensuring most fraudulent transactions are identified

Good precision, reducing unnecessary false alarms

Feature importance analysis shows that:

risk_score

amount

transaction type
are the most influential features

The integration of SMOTE and a rule-based risk score significantly improves the model’s ability to learn fraud patterns.

---

## Conclusion

Detecting mobile money fraud is challenging due to extreme class imbalance and complex transaction behaviors. This project demonstrates that combining domain-driven risk scoring with machine learning leads to improved fraud detection performance. The Random Forest model provides a strong baseline, while SMOTE ensures meaningful learning from rare fraud cases.

Next Steps / Improvements

Experiment with XGBoost or LightGBM for faster training and improved accuracy

Perform hyperparameter tuning on the Random Forest model

Explore real-time fraud detection for streaming transactions

Evaluate cost-sensitive learning approaches

---

## Credits

Dataset: Kaggle – PaySim1

Libraries: Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib
