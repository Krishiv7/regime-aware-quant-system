import pandas as pd
import logging
from ..data.validation import assign_trading_day

logger = logging.getLogger(__name__)

def compute_earnings_features(earnings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes earnings features mapped to the effective trading date.
    """
    if earnings_df.empty:
        return pd.DataFrame()
        
    df = earnings_df.copy()
    
    # Apply no-lookahead adjustment
    df['effective_date'] = df['publication_timestamp'].apply(assign_trading_day)
    
    df = df.sort_values('publication_timestamp').drop_duplicates(subset=['ticker', 'effective_date'], keep='last')
    
    df.set_index(['ticker', 'effective_date'], inplace=True)
    df.index.names = ['Ticker', 'Date']
    
    features = df[['eps_surprise', 'eps_actual', 'revenue_actual']]
    return features
