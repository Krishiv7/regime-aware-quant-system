import logging
from quant_research.config.settings import load_settings
from quant_research.data.market_data import MarketDataProvider
from quant_research.data.news_data import KaggleNewsProvider
from quant_research.data.earnings_data import MockEarningsProvider
from quant_research.features.feature_pipeline import build_feature_pipeline
from quant_research.data.caching import save_to_cache
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    settings = load_settings()
    
    # Load raw data
    mdp = MarketDataProvider(settings.universe)
    market_df = mdp.get_processed_data()
    
    n_provider = KaggleNewsProvider(settings.universe.tickers, settings.universe.start_date, settings.universe.end_date)
    news_df = n_provider.fetch_news()
    
    e_provider = MockEarningsProvider(settings.universe.tickers, settings.universe.start_date, settings.universe.end_date)
    earnings_df = e_provider.fetch_earnings()
    
    # Build features
    # Passing None for nlp_config here in mock mode if you want to skip actual FinBERT inference
    # for speed, but let's pass the config so it runs the actual model if desired.
    # Actually, to make testing fast, we'll leave it as None by default in script unless passed.
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-nlp", action="store_true", help="Run actual FinBERT model on mock news")
    args = parser.parse_args()
    
    nlp_cfg = settings.nlp if args.run_nlp else None
    
    features_df = build_feature_pipeline(market_df, news_df, earnings_df, nlp_config=nlp_cfg)
    
    save_to_cache(features_df, "features_pipeline_output", stage="features")
    logger.info("Features generated and saved successfully.")

if __name__ == "__main__":
    main()
