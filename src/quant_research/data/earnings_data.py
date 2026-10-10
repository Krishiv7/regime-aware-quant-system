import pandas as pd
import numpy as np
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class MockEarningsProvider:
    """
    Mock earnings provider to simulate quarterly earnings releases.
    """
    def __init__(self, tickers: list[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        np.random.seed(42)
        
    def fetch_earnings(self) -> pd.DataFrame:
        logger.info("Generating mock earnings data")
        
        records = []
        for ticker in self.tickers:
            # Generate roughly quarterly dates
            current_date = self.start_date + timedelta(days=np.random.randint(15, 45))
            
            while current_date <= self.end_date:
                # Earnings usually released before market open (08:00) or after close (16:30)
                hour = 8 if np.random.rand() < 0.5 else 16
                minute = 30 if hour == 16 else 0
                publish_time = current_date.replace(hour=hour, minute=minute)
                
                eps_estimate = round(np.random.uniform(0.5, 5.0), 2)
                eps_actual = round(eps_estimate + np.random.normal(0, 0.2), 2)
                surprise = eps_actual - eps_estimate
                
                records.append({
                    "ticker": ticker,
                    "fiscal_period": f"Q{(current_date.month-1)//3 + 1} {current_date.year}",
                    "publication_timestamp": publish_time,
                    "eps_estimate": eps_estimate,
                    "eps_actual": eps_actual,
                    "eps_surprise": surprise,
                    "revenue_actual": round(np.random.uniform(1e9, 1e10), 0)
                })
                
                # Next quarter
                current_date += timedelta(days=90 + np.random.randint(-5, 5))
                
        df = pd.DataFrame(records)
        return df
