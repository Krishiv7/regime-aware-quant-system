import pytest
import pandas as pd
from quant_research.data.validation import assign_trading_day

def test_assign_trading_day():
    # Before close (Tuesday)
    ts1 = pd.Timestamp("2023-10-10 10:00:00")
    assert assign_trading_day(ts1) == pd.Timestamp("2023-10-10")
    
    # After close (Tuesday)
    ts2 = pd.Timestamp("2023-10-10 16:30:00")
    assert assign_trading_day(ts2) == pd.Timestamp("2023-10-11")
    
    # Friday after close -> Monday
    ts3 = pd.Timestamp("2023-10-13 17:00:00") # Friday
    assert assign_trading_day(ts3) == pd.Timestamp("2023-10-16")
    
    # Saturday -> Monday
    ts4 = pd.Timestamp("2023-10-14 12:00:00")
    assert assign_trading_day(ts4) == pd.Timestamp("2023-10-16")
