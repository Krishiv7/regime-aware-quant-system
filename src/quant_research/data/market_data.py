import yfinance as yf
import pandas as pd
import logging
from typing import List, Optional

from .caching import load_from_cache, save_to_cache
from ..config.settings import UniverseConfig

logger = logging.getLogger(__name__)

class MarketDataProvider:
    def __init__(self, config: UniverseConfig):
        self.config = config

    def fetch_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetches historical market data for the configured universe.
        Returns a single-index DataFrame with Date and Ticker in index.
        """
        cache_id = f"market_data_processed_{'_'.join(self.config.tickers)}_{self.config.start_date}_{self.config.end_date}"
        
        if not force_refresh:
            cached_df = load_from_cache(cache_id, stage="processed")
            if cached_df is not None:
                logger.info(f"Loaded market data from cache for {self.config.tickers}")
                return cached_df
                
        logger.info(f"Downloading market data for {self.config.tickers} from {self.config.start_date} to {self.config.end_date}")
        
        try:
            df = yf.download(
                self.config.tickers,
                start=self.config.start_date,
                end=self.config.end_date,
                group_by="ticker",
                auto_adjust=False
            )
        except Exception as e:
            logger.error(f"Failed to download market data: {e}")
            raise
            
        # yfinance returns MultiIndex if multiple tickers, else single level sometimes.
        if not isinstance(df.columns, pd.MultiIndex):
            ticker = self.config.tickers[0]
            df.columns = pd.MultiIndex.from_product([[ticker], df.columns])

            
        # Stack so that we have ['Date', 'Ticker'] as index
        try:
            stacked = df.stack(level=0, future_stack=True)
        except TypeError:
            stacked = df.stack(level=0)
            
        stacked.index.names = ['Date', 'Ticker']
        
        # Ensure timestamp is normalized (localize to UTC or keep naive based on exchange)
        stacked.reset_index(inplace=True)
        stacked['Date'] = pd.to_datetime(stacked['Date']).dt.tz_localize(None)
        stacked.set_index(['Date', 'Ticker'], inplace=True)
        
        stacked.dropna(how='all', inplace=True)
        
        save_to_cache(stacked, cache_id, stage="processed")
        return stacked

    def get_processed_data(self, force_refresh: bool = False) -> pd.DataFrame:
        return self.fetch_data(force_refresh=force_refresh)
