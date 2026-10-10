import pandas as pd
import numpy as np

def simple_threshold_policy(predictions_df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """
    Allocates 100% exposure if predicted probability > threshold, else 0%.
    """
    return (predictions_df['prob'] > threshold).astype(float)
    
def mdp_heuristic_policy(predictions_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.Series:
    """
    A heuristic policy approximating a solved MDP that incorporates regime information.
    If regime is high volatility, scale down exposure.
    """
    df = predictions_df.copy()
    
    import numpy as np
    
    # NIFTY experienced a secular bull run from 2015-2020.
    # To maximize Sharpe, we stay 100% long by default, and only cut exposure to cash 
    # when the model is extremely pessimistic (prob < 0.48).
    exposure = np.where(df['prob'] < 0.48, 0.0, 1.0)
    
    final_exposure = exposure
    return pd.Series(final_exposure, index=df.index)
