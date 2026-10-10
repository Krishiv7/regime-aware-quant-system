import pandas as pd
import requests
import time
import logging

logger = logging.getLogger(__name__)

class KaggleNewsProvider:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_news(self):
        """
        Fetches ACTUAL Indian financial news from Kaggle dataset.
        """
        logger.info(f"Fetching ACTUAL historical news for {self.tickers} from {self.start_date} to {self.end_date} (Kaggle Dataset)")
        
        path = "/Users/krishiv/.cache/kagglehub/datasets/hkapoor/indian-financial-news-articles-20032020/versions/1/IndianFinancialNews.csv"
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.error(f"Could not load kaggle dataset: {e}")
            return pd.DataFrame()
            
        df['publication_timestamp'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['publication_timestamp'])
        
        mask = (df['publication_timestamp'] >= self.start_date) & (df['publication_timestamp'] <= self.end_date)
        df = df[mask]
        
        df['headline'] = df['Title']
        df['article_text'] = df['Description']
        
        final_dfs = []
        for ticker in self.tickers:
            ticker_df = df.copy()
            ticker_df['ticker'] = ticker
            final_dfs.append(ticker_df)
            
        if not final_dfs:
            return pd.DataFrame()
            
        final_df = pd.concat(final_dfs, ignore_index=True)
        final_df = final_df[['publication_timestamp', 'headline', 'article_text', 'ticker']]
        final_df = final_df.sort_values('publication_timestamp').reset_index(drop=True)
        
        logger.info(f"Successfully loaded {len(final_df)} real historical news articles from Kaggle.")
        return final_df
