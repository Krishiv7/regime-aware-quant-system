import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def generate_data_quality_report(market_df: pd.DataFrame, news_df: pd.DataFrame, earnings_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a data quality report before model training.
    """
    logger.info("Generating Data Quality Report")
    
    report = {
        "market_data": {},
        "news_data": {},
        "earnings_data": {}
    }
    
    # Market Data Quality
    report["market_data"]["total_rows"] = len(market_df)
    report["market_data"]["missing_values"] = market_df.isna().sum().to_dict()
    report["market_data"]["tickers"] = list(market_df.index.get_level_values('Ticker').unique())
    if not market_df.empty:
        report["market_data"]["start_date"] = str(market_df.index.get_level_values('Date').min())
        report["market_data"]["end_date"] = str(market_df.index.get_level_values('Date').max())
    
    # News Data Quality
    report["news_data"]["total_rows"] = len(news_df)
    if not news_df.empty:
        report["news_data"]["start_date"] = str(news_df['publication_timestamp'].min())
        report["news_data"]["end_date"] = str(news_df['publication_timestamp'].max())
        report["news_data"]["tickers"] = list(news_df['ticker'].unique())
    
    # Earnings Data Quality
    report["earnings_data"]["total_rows"] = len(earnings_df)
    if not earnings_df.empty:
        report["earnings_data"]["start_date"] = str(earnings_df['publication_timestamp'].min())
        report["earnings_data"]["end_date"] = str(earnings_df['publication_timestamp'].max())
    
    for key, val in report.items():
        logger.info(f"{key.upper()} QUALITY: {val}")
        
    return report

def assign_trading_day(timestamp: pd.Timestamp, market_close_hour: int = 16, market_close_minute: int = 0) -> pd.Timestamp:
    """
    Assigns a publication timestamp to the corresponding effective trading day.
    If published after market close on day T, it becomes available for trading on day T+1.
    Note: Simplification. Proper implementation would use exchange calendars (e.g. pandas-market-calendars)
    to handle weekends and holidays accurately.
    """
    # Check if after market close
    is_after_close = (timestamp.hour > market_close_hour) or \
                     (timestamp.hour == market_close_hour and timestamp.minute >= market_close_minute)
                     
    effective_date = timestamp.normalize()
    if is_after_close:
        effective_date += pd.Timedelta(days=1)
        
    # Move to next weekday if weekend (naive calendar)
    if effective_date.dayofweek == 5: # Saturday
        effective_date += pd.Timedelta(days=2)
    elif effective_date.dayofweek == 6: # Sunday
        effective_date += pd.Timedelta(days=1)
        
    return effective_date
