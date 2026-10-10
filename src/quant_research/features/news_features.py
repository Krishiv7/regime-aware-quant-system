import pandas as pd
import logging
from ..data.validation import assign_trading_day

logger = logging.getLogger(__name__)

def compute_news_features(news_df: pd.DataFrame, nlp_config=None) -> pd.DataFrame:
    """
    Computes daily aggregated news sentiment features per ticker.
    """
    if news_df.empty:
        return pd.DataFrame()
        
    df = news_df.copy()
    
    # Run sentiment analysis if not already present
    if 'sentiment_score' not in df.columns and nlp_config is not None:
        from ..nlp.finbert import FinBERTSentimentAnalyzer
        analyzer = FinBERTSentimentAnalyzer(nlp_config)
        
        logger.info(f"Running FinBERT inference on {len(df)} articles")
        texts = df['headline'].tolist()
        results = analyzer.analyze(texts)
        
        for key in ["prob_positive", "prob_negative", "prob_neutral", "sentiment_score"]:
            df[key] = [r[key] for r in results]
            
    # Mock fallback if NLP is disabled for fast tests
    if 'sentiment_score' not in df.columns:
        import numpy as np
        logger.warning("Using mock sentiment since it is missing and NLP config not provided.")
        df['sentiment_score'] = np.random.uniform(-1, 1, len(df))
        df['prob_positive'] = np.random.uniform(0, 1, len(df))
        df['prob_negative'] = np.random.uniform(0, 1, len(df))
    
    # Apply no-lookahead timestamp adjustment
    df['effective_date'] = df['publication_timestamp'].apply(assign_trading_day)
    
    # Aggregate by Ticker and Effective Date
    agg_df = df.groupby(['ticker', 'effective_date']).agg(
        news_volume=('headline', 'count'),
        avg_sentiment=('sentiment_score', 'mean'),
        sentiment_dispersion=('sentiment_score', 'std'),
        avg_prob_positive=('prob_positive', 'mean'),
        avg_prob_negative=('prob_negative', 'mean')
    )
    
    agg_df['sentiment_dispersion'] = agg_df['sentiment_dispersion'].fillna(0)
    
    # Rename index to match market data convention
    agg_df.index.names = ['Ticker', 'Date']
    
    return agg_df
