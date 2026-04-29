from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

def get_model():
    """Returns the baseline Random Forest configuration."""
    return RandomForestClassifier(
        n_estimators=400, 
        random_state=42, 
        n_jobs=-1
    )

def evaluate_baseline(model, X_test, y_test):
    """Generates the standard performance metrics used in the paper."""
    pred_probs = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    print("Baseline Model: Random Forest + SMOTE")
    print("AUROC:", roc_auc_score(y_test, pred_probs))
    print("AUPRC (Precision-Recall AUC):", average_precision_score(y_test, pred_probs))
    print(classification_report(y_test, y_pred))
