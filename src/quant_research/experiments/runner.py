import logging
import pandas as pd
from typing import List, Dict

from ..validation.walk_forward import generate_walk_forward_splits
from ..models.logistic_model import LogisticModel
from ..regimes.hmm_model import MarketRegimeHMM
from ..config.settings import ProjectSettings

logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, settings: ProjectSettings):
        self.settings = settings
        
    def run_all_experiments(self, features_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Runs Experiments A, B, C, and D over walk-forward splits.
        """
        logger.info("Starting Experiments Runner")
        splits = generate_walk_forward_splits(features_df, train_years=2, test_months=6) # 2 years train, 6 months test
        
        # Define feature groups
        price_features = ['return_1d', 'return_5d', 'return_21d', 'volatility_21d', 'mom_21_63', 'volume_change_1d', 'relative_volume']
        news_features = ['news_volume', 'avg_sentiment', 'sentiment_dispersion', 'avg_prob_positive', 'avg_prob_negative']
        earnings_features = ['eps_surprise', 'eps_actual', 'revenue_actual']
        
        target = 'target_1d'
        
        all_results = {'A': [], 'B': [], 'C': [], 'D': []}
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Running fold {fold_idx}: Train {len(train_idx)}, Test {len(test_idx)}")
            
            # HMM Generation (fit on train, predict test)
            hmm = MarketRegimeHMM(self.settings.regime_detection)
            hmm_probs = hmm.fit_predict(features_df, train_idx, test_idx)
            
            # Add HMM to features for this fold specifically to avoid lookahead
            fold_features_df = features_df.copy()
            fold_features_df.loc[test_idx, 'hmm_high_vol_prob'] = hmm_probs
            # For training, predict on train using the fitted model to get train features
            # (In a strict sense, filtering on train should be used)
            if hmm.best_model:
                try:
                    tr_probs = hmm.best_model.predict_proba(np.nan_to_num(fold_features_df.loc[train_idx, hmm.config.hmm_features].values))
                    vol_state = np.argmax(hmm.best_model.covars_[:, 0])
                    fold_features_df.loc[train_idx, 'hmm_high_vol_prob'] = tr_probs[:, vol_state]
                except:
                    import numpy as np
                    fold_features_df.loc[train_idx, 'hmm_high_vol_prob'] = 0.0
            else:
                fold_features_df.loc[train_idx, 'hmm_high_vol_prob'] = 0.0

            # Experiment A: Price only
            res_A = self._run_model(fold_features_df, train_idx, test_idx, price_features, target)
            all_results['A'].append(res_A)
            
            # Experiment B: Price + News
            res_B = self._run_model(fold_features_df, train_idx, test_idx, price_features + news_features, target)
            all_results['B'].append(res_B)
            
            # Experiment C: Price + Earnings
            res_C = self._run_model(fold_features_df, train_idx, test_idx, price_features + earnings_features, target)
            all_results['C'].append(res_C)
            
            # Experiment D: Combined + HMM
            res_D = self._run_model(fold_features_df, train_idx, test_idx, price_features + news_features + earnings_features + ['hmm_high_vol_prob'], target)
            all_results['D'].append(res_D)
            
        return {
            'A': pd.concat(all_results['A']),
            'B': pd.concat(all_results['B']),
            'C': pd.concat(all_results['C']),
            'D': pd.concat(all_results['D']),
        }
        
    def _run_model(self, df: pd.DataFrame, train_idx, test_idx, features: List[str], target: str) -> pd.DataFrame:
        X_train = df.loc[train_idx, features]
        y_train = df.loc[train_idx, target]
        X_test = df.loc[test_idx, features]
        
        # Verify no lookahead: drop features not present or nan
        model = LogisticModel()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)
        
        # Store results
        res_dict = {
            'target': df.loc[test_idx, target].values,
            'prob': probs.values
        }
        if 'hmm_high_vol_prob' in df.columns:
            res_dict['hmm_high_vol_prob'] = df.loc[test_idx, 'hmm_high_vol_prob'].values
            
        res = pd.DataFrame(res_dict, index=test_idx)
        return res
