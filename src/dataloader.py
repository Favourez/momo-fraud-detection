# src/data_loader.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data(path):
    """
    Load CSV dataset.
    """
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    """
    Preprocess dataset: encode categorical features and split X/y.
    """
    df_processed = df.copy()
    
    # Encode 'type'
    df_processed['type'] = LabelEncoder().fit_transform(df_processed['type'])
    
    # Features and target
    X = df_processed.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)
    y = df_processed['isFraud']
    
    return X, y

def split_and_resample(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train/test and apply SMOTE to training set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    return X_train_res, X_test, y_train_res, y_test
