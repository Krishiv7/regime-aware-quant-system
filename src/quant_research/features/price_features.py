import pandas as pd
import numpy as np

def compute_price_features(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes price and momentum features.
    market_df is expected to have 'Date' and 'Ticker' in index, and OHLCV in columns.
    """
    df = market_df.copy()
    df.sort_index(level=['Ticker', 'Date'], inplace=True)
    
    def calculate_ticker_features(group):
        # Lagged returns
        group['return_1d'] = group['Close'].pct_change(1)
        group['return_5d'] = group['Close'].pct_change(5)
        group['return_21d'] = group['Close'].pct_change(21)
        
        # Volatility
        group['volatility_21d'] = group['return_1d'].rolling(21).std() * np.sqrt(252)
        
        # Momentum
        group['sma_21'] = group['Close'].rolling(21).mean()
        group['sma_63'] = group['Close'].rolling(63).mean()
        group['mom_21_63'] = group['sma_21'] / group['sma_63'] - 1
        
        # Volume
        group['volume_change_1d'] = group['Volume'].pct_change(1)
        group['volume_sma_21'] = group['Volume'].rolling(21).mean()
        group['relative_volume'] = group['Volume'] / group['volume_sma_21']
        
        # Target: 1 if next day return is strictly positive, else 0
        # NOTE: Target is shifted back! So target for Date T is the return from T to T+1
        # This is what we want to predict AT time T.
        next_day_return = group['Close'].pct_change().shift(-1)
        group['target_1d'] = (next_day_return > 0).astype(int)
        
        # Remove target for the last row since we can't observe it yet
        group.loc[group.index[-1], 'target_1d'] = np.nan
        
        return group
        
    df = df.groupby(level='Ticker', group_keys=False).apply(calculate_ticker_features)
    
    return df
