import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss, log_loss

def calculate_classification_metrics(results_df: pd.DataFrame) -> dict:
    y_true = results_df['target']
    probs = results_df['prob']
    y_pred = (probs > 0.5).astype(int)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, probs) if len(y_true.unique()) > 1 else 0.5,
        'brier_score': brier_score_loss(y_true, probs),
        'log_loss': log_loss(y_true, probs) if len(y_true.unique()) > 1 else 0.0
    }
