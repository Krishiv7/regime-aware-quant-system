import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .base import BaseModel

class LogisticModel(BaseModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        self.scaler = StandardScaler()
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        import numpy as np
        X_train_clean = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.fit_transform(X_train_clean)
        self.model.fit(X_scaled, y_train)
        
    def predict_proba(self, X_test: pd.DataFrame) -> pd.Series:
        import numpy as np
        X_test_clean = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.transform(X_test_clean)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return pd.Series(probs, index=X_test.index, name='pred_prob')
        
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        import numpy as np
        X_test_clean = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.transform(X_test_clean)
        preds = self.model.predict(X_scaled)
        return pd.Series(preds, index=X_test.index, name='pred')
