import pandas as pd
from typing import Dict, Any, Optional

class BaseModel:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        raise NotImplementedError
        
    def predict_proba(self, X_test: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
        
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        raise NotImplementedError
