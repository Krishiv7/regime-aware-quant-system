import pandas as pd
import numpy as np

def bootstrap_confidence_interval(metric_func, results_df: pd.DataFrame, n_iterations: int = 1000, alpha: float = 0.05):
    """
    Computes a bootstrap confidence interval for a given metric.
    """
    scores = []
    n_size = len(results_df)
    
    for _ in range(n_iterations):
        # Time-series aware block bootstrap would be better, 
        # but for simplicity we do random resampling here as a placeholder.
        sample = results_df.sample(n=n_size, replace=True)
        # Avoid issues where the sample contains only one class
        if len(sample['target'].unique()) > 1:
            scores.append(metric_func(sample))
            
    if not scores:
        return (0.0, 0.0)
        
    lower = np.percentile(scores, alpha / 2.0 * 100)
    upper = np.percentile(scores, (1 - alpha / 2.0) * 100)
    
    return (lower, upper)
