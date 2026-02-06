# src/visualization.py

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Plots the confusion matrix.
    
    Parameters:
    - y_true: true labels
    - y_pred: predicted labels
    - title: title of the plot
    """
    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap='Blues')
    plt.title(title)
    plt.show()


def plot_feature_importance(model, feature_names, title="Feature Importance"):
    """
    Plots feature importance for tree-based models.
    
    Parameters:
    - model: trained model with feature_importances_ attribute
    - feature_names: list of feature names
    - title: title of the plot
    """
    importances = model.feature_importances_
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, importances)
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()


def plot_top_risk_transactions(transactions, top_n=10, key='risk_score'):
    """
    Plots the top N risky transactions as a bar chart.
    
    Parameters:
    - transactions: list of dictionaries or DataFrame with key column
    - top_n: number of top transactions to plot
    - key: column/key to plot (default 'risk_score')
    """
    # If input is a list of dicts, convert to lists
    if isinstance(transactions, list):
        transactions_sorted = sorted(transactions, key=lambda x: x[key], reverse=True)[:top_n]
        values = [t[key] for t in transactions_sorted]
        labels = [f"{t['type']}_{t['amount']}" for t in transactions_sorted]
    else:
        transactions_sorted = transactions.sort_values(by=key, ascending=False).head(top_n)
        values = transactions_sorted[key].tolist()
        labels = transactions_sorted.index.astype(str).tolist()
    
    plt.figure(figsize=(10, 6))
    plt.barh(labels[::-1], values[::-1], color='orange')
    plt.title(f"Top {top_n} Risky Transactions")
    plt.xlabel(key)
    plt.ylabel("Transaction")
    plt.show()
