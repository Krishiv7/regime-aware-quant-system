import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import logging
from ..config.settings import RegimeConfig

logger = logging.getLogger(__name__)

class MarketRegimeHMM:
    def __init__(self, config: RegimeConfig):
        self.config = config
        self.best_model = None
        
    def fit_predict(self, features_df: pd.DataFrame, train_idx: pd.Index, test_idx: pd.Index) -> pd.Series:
        """
        Fits HMM on train_idx and predicts regime probabilities for test_idx.
        Returns the probability of the highest-volatility regime.
        """
        missing = [f for f in self.config.hmm_features if f not in features_df.columns]
        if missing:
            logger.warning(f"Missing HMM features: {missing}. Filling with zeros.")
            for f in missing:
                features_df[f] = 0.0
                
        X_train = features_df.loc[train_idx, self.config.hmm_features].values
        X_test = features_df.loc[test_idx, self.config.hmm_features].values
        
        # Handle NaNs and Infs in HMM features
        X_train = np.nan_to_num(X_train, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, posinf=0.0, neginf=0.0)
        
        best_score = -np.inf
        
        for n_states in self.config.hmm_n_states:
            try:
                model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=42)
                model.fit(X_train)
                score = model.score(X_train)
                if score > best_score:
                    best_score = score
                    self.best_model = model
            except Exception as e:
                logger.warning(f"HMM fit failed for {n_states} states: {e}")
                
        if self.best_model is None:
            logger.error("All HMM fits failed, returning zeros.")
            return pd.Series(0.0, index=test_idx, name='hmm_high_vol_prob')
            
        # Predict regime probabilities
        probs = self.best_model.predict_proba(X_test)
        
        # Find state with max variance for the first feature (assuming it's 'returns' or similar)
        # covars_ shape is (n_components, n_features, n_features) if full, but (n_components, n_features) if diag.
        if hasattr(self.best_model, 'covars_'):
            vol_state = np.argmax(self.best_model.covars_[:, 0])
            # Ensure bounds
            vol_state = min(vol_state, probs.shape[1] - 1)
        else:
            vol_state = probs.shape[1] - 1
            
        regime_prob = probs[:, vol_state] if probs.ndim == 2 else probs
        return pd.Series(regime_prob, index=test_idx, name='hmm_high_vol_prob')
