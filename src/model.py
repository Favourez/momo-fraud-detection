# src/model.py

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train, random_state=42):
    """
    Train an XGBoost classifier.
    """
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    xgb.fit(X_train, y_train)
    return xgb

def evaluate_model(model, X_test, y_test):
    """
    Predict and print classification report.
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return y_pred
