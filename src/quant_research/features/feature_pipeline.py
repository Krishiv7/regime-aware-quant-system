import pandas as pd
import logging
from typing import Tuple

from .price_features import compute_price_features
from .news_features import compute_news_features
from .earnings_features import compute_earnings_features

logger = logging.getLogger(__name__)

def build_feature_pipeline(market_df: pd.DataFrame, news_df: pd.DataFrame, earnings_df: pd.DataFrame, nlp_config=None) -> pd.DataFrame:
    """
    Merges all features into a single panel DataFrame, ensuring no lookahead bias.
    """
    logger.info("Building feature pipeline")
    
    # 1. Price features
    price_feats = compute_price_features(market_df)
    
    # 2. News features
    news_feats = compute_news_features(news_df, nlp_config)
    
    # 3. Earnings features
    earnings_feats = compute_earnings_features(earnings_df)
    
    # Join everything onto the price DataFrame skeleton
    # Left join ensures we only have rows where the market was open
    df = price_feats.copy()
    
    if not news_feats.empty:
        df = df.join(news_feats, how='left')
        # Missing news means 0 news volume and 0 sentiment
        df['news_volume'] = df['news_volume'].fillna(0)
        df['avg_sentiment'] = df['avg_sentiment'].fillna(0)
        df['sentiment_dispersion'] = df['sentiment_dispersion'].fillna(0)
        df['avg_prob_positive'] = df['avg_prob_positive'].fillna(0)
        df['avg_prob_negative'] = df['avg_prob_negative'].fillna(0)
        
    if not earnings_feats.empty:
        df = df.join(earnings_feats, how='left')
        # Forward fill earnings up to 90 days
        # Sort properly first
        df.sort_index(level=['Ticker', 'Date'], inplace=True)
        earnings_cols = earnings_feats.columns
        for col in earnings_cols:
            df[col] = df.groupby(level='Ticker')[col].ffill(limit=63) # approx 63 trading days in a quarter
            # Fill remaining initial NaNs with 0 (assuming no surprise)
            df[col] = df[col].fillna(0)
            
    # Drop rows where target is NaN (the very last day of data)
    df.dropna(subset=['target_1d'], inplace=True)
    
    # Drop initial rows where rolling windows generated NaNs
    df.dropna(subset=['sma_63'], inplace=True)
    
    logger.info(f"Feature pipeline generated {len(df)} rows across {len(df.index.get_level_values('Ticker').unique())} tickers.")
    return df
