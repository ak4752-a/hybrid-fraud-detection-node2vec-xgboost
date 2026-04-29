import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(path="creditcard.csv"):
    """Loads the Credit Card dataset and separates features from the target."""
    df = pd.read_csv(path)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y

def process_and_balance(X_train, y_train, X_test):
    """
    Applies SMOTE to balance the training set and normalizes all features 
    using Standard Scaling.
    """
    # Resample training data only to prevent leakage
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
    
    # Scale features
    scaler = StandardScaler()
    X_train_bal = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_bal, y_train_bal, X_test_scaled
