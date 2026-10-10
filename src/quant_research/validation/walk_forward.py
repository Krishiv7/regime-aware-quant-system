import pandas as pd
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def generate_walk_forward_splits(df: pd.DataFrame, train_years: int = 3, test_months: int = 6) -> List[Tuple[pd.Index, pd.Index]]:
    """
    Generates chronological train/test splits, strictly preventing lookahead.
    """
    dates = df.index.get_level_values('Date').unique().sort_values()
    if len(dates) == 0:
        return []
        
    start_date = dates.min()
    end_date = dates.max()
    
    splits = []
    current_train_end = start_date + pd.DateOffset(years=train_years)
    
    while current_train_end < end_date:
        current_test_end = min(current_train_end + pd.DateOffset(months=test_months), end_date)
        
        train_mask = df.index.get_level_values('Date') < current_train_end
        test_mask = (df.index.get_level_values('Date') >= current_train_end) & (df.index.get_level_values('Date') < current_test_end)
        
        train_idx = df[train_mask].index
        test_idx = df[test_mask].index
                      
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
            
        current_train_end = current_test_end
        
    return splits
