import pandas as pd
import logging
import json
from pathlib import Path
from quant_research.data.caching import load_from_cache
from quant_research.backtesting.metrics import calculate_classification_metrics
from quant_research.validation.statistical_tests import bootstrap_confidence_interval
from quant_research.backtesting.engine import BacktestEngine
from quant_research.portfolio.policies import simple_threshold_policy, mdp_heuristic_policy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_report():
    logger.info("Generating Final Report")
    
    features_df = load_from_cache("features_pipeline_output", stage="features")
    
    experiments = ['A', 'B', 'C', 'D']
    metrics_summary = {}
    
    engine = BacktestEngine(initial_capital=100000, tc_bps=5.0)
    portfolio_summary = {}
    
    for exp in experiments:
        preds = load_from_cache(f"experiment_{exp}_predictions", stage="processed")
        if preds is None:
            logger.warning(f"Skipping {exp}, predictions not found.")
            continue
            
        # 1. Predictive Metrics
        metrics = calculate_classification_metrics(preds)
        ci_lower, ci_upper = bootstrap_confidence_interval(
            lambda df: calculate_classification_metrics(df)['accuracy'], 
            preds, 
            n_iterations=100
        )
        metrics['accuracy_ci_95'] = [ci_lower, ci_upper]
        metrics_summary[exp] = metrics
        
        # 2. Portfolio Backtest
        # Use simple policy for A, B, C. Use MDP policy for D.
        if exp == 'D':
            policy = mdp_heuristic_policy(preds, features_df)
        else:
            policy = simple_threshold_policy(preds)
            
        bt_results = engine.run(features_df, preds, policy)
        
        annualized_ret = bt_results['net_return'].mean() * 252
        annualized_vol = bt_results['net_return'].std() * (252 ** 0.5)
        sharpe = annualized_ret / annualized_vol if annualized_vol > 0 else 0
        max_dd = bt_results['drawdown'].min()
        
        portfolio_summary[exp] = {
            'annualized_return': annualized_ret,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd
        }
        
    report = {
        'predictive_metrics': metrics_summary,
        'portfolio_performance': portfolio_summary
    }
    
    out_dir = Path("reports/experiment_summaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "final_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info("Report saved to reports/experiment_summaries/final_report.json")
    for exp, met in metrics_summary.items():
        logger.info(f"Exp {exp} Acc: {met['accuracy']:.4f}  Sharpe: {portfolio_summary[exp]['sharpe_ratio']:.2f}")

if __name__ == "__main__":
    generate_report()
