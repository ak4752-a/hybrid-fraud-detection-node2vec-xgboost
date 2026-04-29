from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier

def get_baseline_rf():
    """Baseline Random Forest setup[cite: 89]."""
    return RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)

def get_balanced_rf():
    """Balanced Random Forest for Hybrid-1[cite: 90]."""
    return BalancedRandomForestClassifier(n_estimators=800, random_state=42, n_jobs=-1)

def get_hybrid_xgboost(y_train):
    """Proposed XGBoost with tuned hyperparameters[cite: 94, 107]."""
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    return XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        random_state=42
    )
