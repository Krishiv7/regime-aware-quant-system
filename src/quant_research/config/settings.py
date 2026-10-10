import yaml
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path

class UniverseConfig(BaseModel):
    market: str = "US"
    country: str = "USA"
    exchange: str = "NASDAQ"
    tickers: List[str] = ["AAPL"]
    start_date: str = "2018-01-01"
    end_date: str = "2023-12-31"
    prediction_horizon_days: int = 1
    trading_calendar: str = "NYSE"

class DataConfig(BaseModel):
    market_data_provider: str = "yfinance"
    news_provider: str = "mock"
    earnings_provider: str = "mock"

class NLPConfig(BaseModel):
    model_name: str = "ProsusAI/finbert"
    batch_size: int = 16

class RegimeConfig(BaseModel):
    hmm_n_states: List[int] = [2, 3]
    hmm_features: List[str] = ["returns", "volatility"]

class ModelsConfig(BaseModel):
    baseline: str = "logistic_regression"

class PortfolioConfig(BaseModel):
    initial_capital: float = 100000.0
    transaction_cost_bps: float = 5.0

class ProjectSettings(BaseModel):
    universe: UniverseConfig
    data: DataConfig
    nlp: NLPConfig
    regime_detection: RegimeConfig
    models: ModelsConfig
    portfolio: PortfolioConfig

def load_settings(config_path: str = "configs/default.yaml") -> ProjectSettings:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    return ProjectSettings(**config_dict)
