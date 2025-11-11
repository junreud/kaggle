"""
Position Mapping Test Script

This script tests position mapping strategies using OOF predictions.
Tests all three strategies:
1. Sharpe Scaling
2. Quantile Binning  
3. Volatility Targeting

Also validates constraints and optimizes parameters.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict
from src.position import (
    SharpeScalingMapper,
    QuantileBinningMapper,
    VolatilityTargetingMapper,
    PositionOptimizer,
    create_position_mapper
)
from src.metric import CompetitionMetric
from src.utils import get_logger, Timer
import warnings
warnings.filterwarnings('ignore')

# Logger
logger = get_logger(log_file="logs/position_test.log", level="INFO")


def load_oof_data() -> pd.DataFrame:
    """Load OOF predictions and actual returns."""
    logger.info("Loading OOF data...")
    
    # Load train data with actual returns
    train_path = Path("data/raw/train.csv")
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found: {train_path}")
    
    df_train = pd.read_csv(train_path)
    logger.info(f"Loaded train data: {len(df_train)} samples")
    
    # Load return predictions (try parquet first, then CSV)
    oof_r_path_parquet = Path("artifacts/oof_r_hat.parquet")
    oof_r_path_csv = Path("artifacts/oof_r_hat.csv")
    
    if oof_r_path_parquet.exists():
        df_r = pd.read_parquet(oof_r_path_parquet)
    elif oof_r_path_csv.exists():
        df_r = pd.read_csv(oof_r_path_csv)
        # Add index as date_id if not present
        if 'date_id' not in df_r.columns:
            df_r['date_id'] = range(len(df_r))
        # Rename oof_prediction to r_hat if needed
        if 'oof_prediction' in df_r.columns:
            df_r = df_r.rename(columns={'oof_prediction': 'r_hat'})
    else:
        raise FileNotFoundError(f"OOF return predictions not found")
    
    # Load risk predictions
    oof_sigma_path = Path("artifacts/oof_sigma_hat.parquet")
    if not oof_sigma_path.exists():
        raise FileNotFoundError(f"OOF risk predictions not found: {oof_sigma_path}")
    
    df_sigma = pd.read_parquet(oof_sigma_path)
    
    # Ensure train data has date_id
    if 'date_id' not in df_train.columns:
        df_train['date_id'] = range(len(df_train))
    
    # Merge all data
    df = df_r.merge(df_sigma[['date_id', 'sigma_hat']], on='date_id', how='inner')
    df = df.merge(df_train[['date_id', 'forward_returns']], on='date_id', how='inner')
    
    # Filter only OOF samples if is_oof column exists
    if 'is_oof' in df.columns:
        df = df[df['is_oof'] == True].copy()
        logger.info(f"Filtered to {len(df)} OOF samples")
    
    # Drop NaN values
    df = df.dropna(subset=['r_hat', 'sigma_hat', 'forward_returns'])
    
    logger.info(f"Final data: {len(df)} samples")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"\nData summary:")
    logger.info(f"  r_hat: mean={df['r_hat'].mean():.6f}, std={df['r_hat'].std():.6f}")
    logger.info(f"  sigma_hat: mean={df['sigma_hat'].mean():.6f}, std={df['sigma_hat'].std():.6f}") 
    logger.info(f"  forward_returns: mean={df['forward_returns'].mean():.6f}, std={df['forward_returns'].std():.6f}")
    
    return df


def test_position_strategy(
    strategy_name: str,
    df: pd.DataFrame,
    optimize: bool = False
) -> Dict:
    """
    Test a position mapping strategy.
    
    Parameters
    ----------
    strategy_name : str
        Strategy name
    df : pd.DataFrame
        OOF data with predictions and actual returns
    optimize : bool
        Whether to optimize parameters
        
    Returns
    -------
    dict
        Results
    """
    logger.info("\n" + "="*80)
    logger.info(f"Testing Strategy: {strategy_name.upper()}")
    logger.info("="*80)
    
    # Create mapper
    mapper = create_position_mapper(strategy_name)
    
    # Extract data
    r_hat = df['r_hat'].values
    sigma_hat = df['sigma_hat'].values
    actual_returns = df['forward_returns'].values
    
    # Remove NaN values
    mask = ~(np.isnan(r_hat) | np.isnan(sigma_hat) | np.isnan(actual_returns))
    r_hat = r_hat[mask]
    sigma_hat = sigma_hat[mask]
    actual_returns = actual_returns[mask]
    
    logger.info(f"Valid samples: {len(r_hat)}")
    
    # Map positions (default parameters)
    logger.info("\n--- Default Parameters ---")
    with Timer("Position mapping"):
        positions = mapper.map_positions(r_hat, sigma_hat)
    
    # Calculate strategy returns
    strategy_returns = positions * actual_returns
    
    # Calculate metrics
    metric_calc = CompetitionMetric()
    result = metric_calc.calculate_score(
        allocations=positions,
        forward_returns=actual_returns
    )
    default_score = result['score']
    
    logger.info(f"\nDefault Score: {default_score:.6f}")
    logger.info(f"Position Stats:")
    logger.info(f"  Mean: {np.mean(positions):.4f}")
    logger.info(f"  Std: {np.std(positions):.4f}")
    logger.info(f"  Min: {np.min(positions):.4f}")
    logger.info(f"  Max: {np.max(positions):.4f}")
    logger.info(f"  Median: {np.median(positions):.4f}")
    
    # Validate constraints
    constraints = mapper.validate_constraints(
        positions=positions,
        actual_returns=actual_returns,
        market_returns=actual_returns
    )
    
    results = {
        'strategy': strategy_name,
        'default_score': default_score,
        'default_positions': positions,
        'constraints': constraints
    }
    
    # Optimize parameters if requested
    if optimize:
        logger.info("\n--- Optimizing Parameters ---")
        optimizer = PositionOptimizer(mapper)
        
        if strategy_name == 'sharpe_scaling':
            optimal_params = optimizer.optimize_sharpe_params(
                r_hat, sigma_hat, actual_returns
            )
            
            # Re-map with optimal parameters
            optimal_positions = mapper.map_positions(
                r_hat, sigma_hat,
                k=optimal_params['k'],
                b=optimal_params['b']
            )
            
            results['optimal_params'] = optimal_params
            results['optimal_positions'] = optimal_positions
            
        elif strategy_name == 'quantile':
            optimal_params = optimizer.optimize_quantile_allocations(
                r_hat, sigma_hat, actual_returns
            )
            
            # Re-map with optimal allocations
            optimal_positions = mapper.map_positions(
                r_hat, sigma_hat,
                allocations=optimal_params['allocations']
            )
            
            results['optimal_params'] = optimal_params
            results['optimal_positions'] = optimal_positions
    
    return results


def compare_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare all strategies.
    
    Parameters
    ----------
    df : pd.DataFrame
        OOF data
        
    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    logger.info("\n" + "="*80)
    logger.info("Strategy Comparison")
    logger.info("="*80)
    
    strategies = ['sharpe_scaling', 'quantile', 'vol_targeting']
    comparison_data = []
    
    for strategy in strategies:
        results = test_position_strategy(strategy, df, optimize=False)
        
        comparison_data.append({
            'strategy': strategy,
            'score': results['default_score'],
            'vol_ratio': results['constraints']['vol_ratio'],
            'vol_ok': results['constraints']['vol_ratio_ok'],
            'leverage_pct': results['constraints']['leverage_pct'],
            'leverage_ok': results['constraints']['leverage_ok'],
            'all_ok': results['constraints']['all_constraints_ok']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    logger.info("\n" + "="*80)
    logger.info("Comparison Summary")
    logger.info("="*80)
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    
    # Save comparison
    output_path = Path("results/position_strategy_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Comparison saved to {output_path}")
    
    return comparison_df


def main():
    """Main test function."""
    logger.info("="*80)
    logger.info("Position Mapping Test")
    logger.info("="*80)
    
    # Load OOF data
    df = load_oof_data()
    
    # Test Sharpe Scaling with optimization
    logger.info("\n" + "#"*80)
    logger.info("# SHARPE SCALING STRATEGY")
    logger.info("#"*80)
    sharpe_results = test_position_strategy('sharpe_scaling', df, optimize=True)
    
    # Test Quantile Binning with optimization
    logger.info("\n" + "#"*80)
    logger.info("# QUANTILE BINNING STRATEGY")
    logger.info("#"*80)
    quantile_results = test_position_strategy('quantile', df, optimize=True)
    
    # Test Volatility Targeting
    logger.info("\n" + "#"*80)
    logger.info("# VOLATILITY TARGETING STRATEGY")
    logger.info("#"*80)
    vol_results = test_position_strategy('vol_targeting', df, optimize=False)
    
    # Compare strategies
    logger.info("\n" + "#"*80)
    logger.info("# STRATEGY COMPARISON")
    logger.info("#"*80)
    comparison_df = compare_strategies(df)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("Test Summary")
    logger.info("="*80)
    
    logger.info("\n✅ Completed Tests:")
    logger.info("  1. Sharpe Scaling (with optimization)")
    logger.info("  2. Quantile Binning (with optimization)")
    logger.info("  3. Volatility Targeting")
    logger.info("  4. Strategy Comparison")
    
    logger.info("\n📁 Results saved to:")
    logger.info("  - results/position_strategy_comparison.csv")
    
    # Best strategy
    best_strategy = comparison_df.loc[comparison_df['score'].idxmax()]
    logger.info("\n🏆 Best Strategy:")
    logger.info(f"  Strategy: {best_strategy['strategy']}")
    logger.info(f"  Score: {best_strategy['score']:.6f}")
    logger.info(f"  Vol Ratio: {best_strategy['vol_ratio']:.4f}")
    logger.info(f"  Constraints OK: {best_strategy['all_ok']}")
    
    if 'optimal_params' in sharpe_results:
        logger.info("\n🎯 Optimal Sharpe Scaling Parameters:")
        logger.info(f"  k: {sharpe_results['optimal_params']['k']:.4f}")
        logger.info(f"  b: {sharpe_results['optimal_params']['b']:.4f}")
        logger.info(f"  Score: {sharpe_results['optimal_params']['score']:.6f}")
    
    if 'optimal_params' in quantile_results:
        logger.info("\n🎯 Optimal Quantile Allocations:")
        logger.info(f"  {quantile_results['optimal_params']['allocations']}")
        logger.info(f"  Score: {quantile_results['optimal_params']['score']:.6f}")
    
    logger.info("\n" + "="*80)
    logger.info("Position Mapping Test Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
