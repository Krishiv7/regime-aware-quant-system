import logging
from quant_research.config.settings import load_settings
from quant_research.data.market_data import MarketDataProvider
from quant_research.data.news_data import KaggleNewsProvider
from quant_research.data.earnings_data import MockEarningsProvider
from quant_research.data.validation import generate_data_quality_report
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force redownload of data")
    args = parser.parse_args()
    
    settings = load_settings()
    
    # Market Data
    mdp = MarketDataProvider(settings.universe)
    market_df = mdp.get_processed_data(force_refresh=args.force)
    
    # News Data
    n_provider = KaggleNewsProvider(
        settings.universe.tickers, 
        settings.universe.start_date, 
        settings.universe.end_date
    )
    news_df = n_provider.fetch_news()
    
    # Earnings Data
    e_provider = MockEarningsProvider(
        settings.universe.tickers,
        settings.universe.start_date,
        settings.universe.end_date
    )
    earnings_df = e_provider.fetch_earnings()
    
    # Quality Report
    report = generate_data_quality_report(market_df, news_df, earnings_df)
    logger.info("Data download and validation complete.")

if __name__ == "__main__":
    main()
