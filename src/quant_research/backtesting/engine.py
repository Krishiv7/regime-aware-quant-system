import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0, tc_bps: float = 5.0):
        self.initial_capital = initial_capital
        self.tc = tc_bps / 10000.0
        
    def run(self, features_df: pd.DataFrame, predictions_df: pd.DataFrame, exposure_policy: pd.Series) -> pd.DataFrame:
        """
        Runs a chronological backtest.
        exposure_policy is a Series with matching index containing exposure values [0, 1].
        """
        # Join predictions and features to get forward returns
        df = features_df[['return_1d']].copy()
        df = df.join(predictions_df, how='inner')
        
        df['exposure'] = exposure_policy
        
        # Forward return (return at T+1 is the return from T to T+1)
        # We assume we enter the position at Close of T, and exit at Close of T+1
        df['next_return'] = df.groupby('Ticker')['return_1d'].shift(-1)
        
        # Drop the last row since we don't know the forward return
        df.dropna(subset=['next_return'], inplace=True)
        
        # Gross return
        df['gross_ret'] = df['exposure'] * df['next_return']
        
        # Transaction costs
        df['turnover'] = df.groupby('Ticker')['exposure'].diff().fillna(df['exposure']).abs()
        df['tc_impact'] = df['turnover'] * self.tc
        
        df['net_ret'] = df['gross_ret'] - df['tc_impact']
        
        # Aggregate across tickers
        port_ret = df.groupby('Date')['net_ret'].sum()
        
        # Equity curve
        equity = self.initial_capital * (1 + port_ret).cumprod()
        
        # Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        
        return pd.DataFrame({
            'net_return': port_ret,
            'equity': equity,
            'drawdown': drawdown,
            'turnover': df.groupby('Date')['turnover'].sum()
        })
