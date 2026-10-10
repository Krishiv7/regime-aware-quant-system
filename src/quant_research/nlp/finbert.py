import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import logging
from ..config.settings import NLPConfig

logger = logging.getLogger(__name__)

class FinBERTSentimentAnalyzer:
    def __init__(self, config: NLPConfig):
        self.config = config
        self.device = torch.device("cpu") # Constraint: Prefer CPU-compatible
        logger.info(f"Loading FinBERT model: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

    def analyze(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Runs batch inference on a list of texts and returns sentiment probabilities.
        """
        if not texts:
            return []
            
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            for prob in probs:
                # Assuming standard ProsusAI/finbert mapping: 0: positive, 1: negative, 2: neutral
                p, n, neu = prob[0].item(), prob[1].item(), prob[2].item()
                
                results.append({
                    "prob_positive": p,
                    "prob_negative": n,
                    "prob_neutral": neu,
                    "sentiment_score": p - n
                })
                
        return results
