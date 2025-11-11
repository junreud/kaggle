"""
Test backtest simulator with transaction costs.

This script demonstrates:
1. Transaction cost impact on different strategies
2. Forward-looking bias detection
3. Detailed backtest reports
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.data import DataLoader
from src.backtest import BacktestSimulator, create_backtest_simulator
from src.utils import get_logger, Timer, set_seed

# Initialize logger
logger = get_logger(log_file="logs/backtest_test.log", level="INFO")


def test_backtest(config_path: str = "conf/params.yaml"):
    """
    Test backtest simulator.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    """
    logger.info("="*80)
    logger.info("Testing Backtest Simulator")
    logger.info("="*80)
    
    # Set random seed
    set_seed(42)
    
    # Load data
    with Timer("Loading data", logger=logger):
        data_loader = DataLoader(config_path)
        train_df, _ = data_loader.load_data()
        logger.info(f"Train data shape: {train_df.shape}")
    
    # Extract required columns
    forward_returns = train_df['forward_returns'].values
    dates = train_df['date_id'].values
    risk_free_rate = train_df['risk_free_rate'].values if 'risk_free_rate' in train_df.columns else None
    
    # Remove NaN values
    valid_mask = ~np.isnan(forward_returns)
    forward_returns_clean = forward_returns[valid_mask]
    dates_clean = dates[valid_mask]
    if risk_free_rate is not None:
        risk_free_rate_clean = risk_free_rate[valid_mask]
    else:
        risk_free_rate_clean = None
    
    logger.info(f"Valid samples: {len(forward_returns_clean)}")
    
    # Initialize backtest simulator
    simulator = create_backtest_simulator(config_path)
    
    # Test 1: Buy and Hold (allocation = 1.0, no turnover)
    logger.info("\n" + "="*80)
    logger.info("Test 1: Buy and Hold Strategy (allocation = 1.0)")
    logger.info("="*80)
    
    allocations_bh = np.ones_like(forward_returns_clean)
    
    results_bh = simulator.run_backtest(
        allocations_bh,
        forward_returns_clean,
        dates=dates_clean,
        risk_free_rate=risk_free_rate_clean,
        apply_costs=True
    )
    
    # Generate report
    report_bh = simulator.generate_report(
        results_bh,
        output_path="results/backtest/buy_hold_report.csv"
    )
    logger.info(f"\n{report_bh.to_string(index=False)}")
    
    # Test 2: High Turnover Strategy
    logger.info("\n" + "="*80)
    logger.info("Test 2: High Turnover Strategy (random allocations)")
    logger.info("="*80)
    
    # Create random allocations that change frequently
    np.random.seed(42)
    n_samples = len(forward_returns_clean)
    allocations_ht = np.random.uniform(0.5, 1.5, n_samples)
    # Smooth a bit to make it realistic
    for i in range(1, n_samples):
        allocations_ht[i] = 0.7 * allocations_ht[i] + 0.3 * allocations_ht[i-1]
    
    results_ht = simulator.run_backtest(
        allocations_ht,
        forward_returns_clean,
        dates=dates_clean,
        risk_free_rate=risk_free_rate_clean,
        apply_costs=True
    )
    
    report_ht = simulator.generate_report(
        results_ht,
        output_path="results/backtest/high_turnover_report.csv"
    )
    logger.info(f"\n{report_ht.to_string(index=False)}")
    
    # Test 3: Momentum Strategy (changes with market)
    logger.info("\n" + "="*80)
    logger.info("Test 3: Simple Momentum Strategy")
    logger.info("="*80)
    
    # Allocation based on rolling mean return (simple momentum)
    window = 20
    rolling_mean = pd.Series(forward_returns_clean).rolling(window, min_periods=1).mean()
    # Scale to allocation: positive -> higher allocation, negative -> lower
    allocations_mom = 1.0 + np.tanh(rolling_mean * 50)  # Scale to roughly 0-2
    allocations_mom = np.clip(allocations_mom, 0, 2)
    
    results_mom = simulator.run_backtest(
        allocations_mom,
        forward_returns_clean,
        dates=dates_clean,
        risk_free_rate=risk_free_rate_clean,
        apply_costs=True
    )
    
    report_mom = simulator.generate_report(
        results_mom,
        output_path="results/backtest/momentum_report.csv"
    )
    logger.info(f"\n{report_mom.to_string(index=False)}")
    
    # Test 4: Forward-looking bias detection
    logger.info("\n" + "="*80)
    logger.info("Test 4: Forward-Looking Bias Detection")
    logger.info("="*80)
    
    # Create a DataFrame with predictions
    test_df = pd.DataFrame({
        'date_id': dates_clean,
        'forward_returns': forward_returns_clean,
        'prediction': rolling_mean  # Using momentum as "prediction"
    })
    
    bias_result = simulator.check_forward_looking_bias(test_df)
    
    logger.info("\nBias Detection Results:")
    logger.info(f"  Bias detected: {bias_result['bias_detected']}")
    logger.info(f"  Prediction rate: {bias_result['prediction_rate']:.2%}")
    logger.info(f"  Correlation t+0: {bias_result['corr_t0']:.4f}")
    logger.info(f"  Correlation t+5: {bias_result['corr_t5']:.4f}")
    logger.info(f"  Correlation t+10: {bias_result['corr_t10']:.4f}")
    
    # Test 5: Create a biased prediction (using future info)
    logger.info("\n" + "="*80)
    logger.info("Test 5: Testing with BIASED predictions (should detect)")
    logger.info("="*80)
    
    # Create obviously biased prediction using future returns
    biased_prediction = pd.Series(forward_returns_clean).shift(-5).fillna(0)
    
    test_df_biased = pd.DataFrame({
        'date_id': dates_clean,
        'forward_returns': forward_returns_clean,
        'prediction': biased_prediction
    })
    
    bias_result_biased = simulator.check_forward_looking_bias(test_df_biased)
    
    logger.info("\nBiased Prediction Detection Results:")
    logger.info(f"  Bias detected: {bias_result_biased['bias_detected']}")
    logger.info(f"  Correlation t+0: {bias_result_biased['corr_t0']:.4f}")
    logger.info(f"  Correlation t+5: {bias_result_biased['corr_t5']:.4f}")
    logger.info(f"  Correlation t+10: {bias_result_biased['corr_t10']:.4f}")
    
    # Compare strategies
    logger.info("\n" + "="*80)
    logger.info("Strategy Comparison Summary")
    logger.info("="*80)
    
    comparison = pd.DataFrame({
        'Strategy': ['Buy & Hold', 'High Turnover', 'Momentum'],
        'Score_Gross': [
            results_bh['gross_metrics']['score'],
            results_ht['gross_metrics']['score'],
            results_mom['gross_metrics']['score']
        ],
        'Score_Net': [
            results_bh['net_metrics']['score'],
            results_ht['net_metrics']['score'],
            results_mom['net_metrics']['score']
        ],
        'Cost_Impact': [
            results_bh['cost_impact']['score_impact'] if results_bh['cost_impact'] else 0,
            results_ht['cost_impact']['score_impact'] if results_ht['cost_impact'] else 0,
            results_mom['cost_impact']['score_impact'] if results_mom['cost_impact'] else 0
        ],
        'Turnover': [
            results_bh['additional_metrics']['turnover'],
            results_ht['additional_metrics']['turnover'],
            results_mom['additional_metrics']['turnover']
        ],
        'N_Trades': [
            results_bh['cost_impact']['n_trades'] if results_bh['cost_impact'] else 0,
            results_ht['cost_impact']['n_trades'] if results_ht['cost_impact'] else 0,
            results_mom['cost_impact']['n_trades'] if results_mom['cost_impact'] else 0
        ]
    })
    
    logger.info(f"\n{comparison.to_string(index=False)}")
    
    # Save comparison
    output_dir = Path("results/backtest")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "strategy_comparison.csv", index=False)
    
    logger.info(f"\nComparison saved to {output_dir / 'strategy_comparison.csv'}")
    
    logger.info("\n" + "="*80)
    logger.info("Backtest Testing Complete")
    logger.info("="*80)


if __name__ == "__main__":
    test_backtest()
