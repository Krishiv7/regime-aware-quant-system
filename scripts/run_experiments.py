import logging
from quant_research.config.settings import load_settings
from quant_research.data.caching import load_from_cache, save_to_cache
from quant_research.experiments.runner import ExperimentRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    settings = load_settings()
    
    features_df = load_from_cache("features_pipeline_output", stage="features")
    if features_df is None:
        logger.error("Features not found. Run generate_features.py first.")
        return
        
    runner = ExperimentRunner(settings)
    results = runner.run_all_experiments(features_df)
    
    for exp_name, res_df in results.items():
        save_to_cache(res_df, f"experiment_{exp_name}_predictions", stage="processed")
        logger.info(f"Experiment {exp_name} completed and saved. Total test predictions: {len(res_df)}")

if __name__ == "__main__":
    main()
