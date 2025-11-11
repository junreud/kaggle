"""
Test feature selection functionality.

This script demonstrates:
1. Feature importance analysis
2. Feature selection by correlation
3. Removing highly correlated features
4. Comparing different selection methods
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.data import DataLoader
from src.features import FeatureEngineering, create_feature_engineering
from src.utils import get_logger, Timer, set_seed

# Initialize logger
logger = get_logger(log_file="logs/feature_selection_test.log", level="INFO")


def test_feature_selection():
    """Test feature selection methods."""
    logger.info("="*80)
    logger.info("Testing Feature Selection")
    logger.info("="*80)
    
    # Set seed
    set_seed(42)
    
    # Load data
    logger.info("\nLoading data...")
    with Timer("Loading data", logger=logger):
        config_path = "conf/params.yaml"
        data_loader = DataLoader(config_path)
        train_data, _ = data_loader.load_data()
        
        logger.info(f"Train data shape: {train_data.shape}")
    
    # Initialize feature engineering
    logger.info("\nInitializing feature engineering...")
    feature_eng = create_feature_engineering(config_path)
    
    # Create features
    logger.info("\nCreating features...")
    train_features = feature_eng.fit_transform(train_data, date_col='date_id')
    
    logger.info(f"\nData shape after feature engineering: {train_features.shape}")
    
    # Test 1: Correlation-based selection (top 100)
    logger.info("\n" + "="*80)
    logger.info("Test 1: Correlation-based Selection (Top 100)")
    logger.info("="*80)
    
    df_selected_corr, selected_corr = feature_eng.select_features_by_importance(
        train_features,
        target_col='forward_returns',
        method='correlation',
        top_n=100
    )
    
    logger.info(f"\nSelected {len(selected_corr)} features")
    logger.info(f"Shape: {df_selected_corr.shape}")
    
    # Test 2: Variance-based selection
    logger.info("\n" + "="*80)
    logger.info("Test 2: Variance-based Selection (Top 100)")
    logger.info("="*80)
    
    df_selected_var, selected_var = feature_eng.select_features_by_importance(
        train_features,
        target_col='forward_returns',
        method='variance',
        top_n=100
    )
    
    logger.info(f"\nSelected {len(selected_var)} features")
    logger.info(f"Shape: {df_selected_var.shape}")
    
    # Test 3: Threshold-based selection
    logger.info("\n" + "="*80)
    logger.info("Test 3: Correlation Threshold Selection (>0.02)")
    logger.info("="*80)
    
    df_selected_thresh, selected_thresh = feature_eng.select_features_by_importance(
        train_features,
        target_col='forward_returns',
        method='correlation',
        threshold=0.02
    )
    
    logger.info(f"\nSelected {len(selected_thresh)} features above threshold")
    logger.info(f"Shape: {df_selected_thresh.shape}")
    
    # Test 4: Remove correlated features
    logger.info("\n" + "="*80)
    logger.info("Test 4: Remove Highly Correlated Features")
    logger.info("="*80)
    
    # First select top features, then remove correlations
    df_filtered, removed_features = feature_eng.remove_correlated_features(
        df_selected_corr,
        threshold=0.95,
        target_col='forward_returns'
    )
    
    logger.info(f"\nRemoved {len(removed_features)} correlated features")
    logger.info(f"Final shape: {df_filtered.shape}")
    
    # Compare feature sets
    logger.info("\n" + "="*80)
    logger.info("Feature Set Comparison")
    logger.info("="*80)
    
    comparison = pd.DataFrame({
        'Method': [
            'Original',
            'Correlation (Top 100)',
            'Variance (Top 100)',
            'Correlation (>0.02)',
            'After Correlation Filter'
        ],
        'N_Features': [
            train_features.shape[1] - 2,  # Exclude date and target
            len(selected_corr),
            len(selected_var),
            len(selected_thresh),
            df_filtered.shape[1] - 2
        ]
    })
    
    logger.info(f"\n{comparison.to_string(index=False)}")
    
    # Overlap analysis
    logger.info("\n" + "="*80)
    logger.info("Feature Overlap Analysis")
    logger.info("="*80)
    
    corr_set = set(selected_corr)
    var_set = set(selected_var)
    
    overlap = corr_set & var_set
    corr_only = corr_set - var_set
    var_only = var_set - corr_set
    
    logger.info(f"\nCorrelation method: {len(corr_set)} features")
    logger.info(f"Variance method: {len(var_set)} features")
    logger.info(f"Overlap: {len(overlap)} features ({len(overlap)/len(corr_set)*100:.1f}%)")
    logger.info(f"Correlation only: {len(corr_only)} features")
    logger.info(f"Variance only: {len(var_only)} features")
    
    # Save results
    logger.info("\n" + "="*80)
    logger.info("Saving Results")
    logger.info("="*80)
    
    output_dir = Path("results/feature_selection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save selected features
    selected_features_df = pd.DataFrame({
        'correlation_top100': pd.Series(selected_corr),
        'variance_top100': pd.Series(selected_var),
        'correlation_threshold': pd.Series(selected_thresh)
    })
    
    output_path = output_dir / "selected_features.csv"
    selected_features_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Selected features saved to {output_path}")
    
    # Save filtered data (use CSV instead of parquet)
    filtered_path = output_dir / "train_features_selected.csv"
    df_filtered.to_csv(filtered_path, index=False)
    logger.info(f"✓ Filtered data saved to {filtered_path}")
    
    # Save comparison
    comparison_path = output_dir / "feature_selection_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    logger.info(f"✓ Comparison saved to {comparison_path}")
    
    logger.info("\n" + "="*80)
    logger.info("Feature Selection Test Complete")
    logger.info("="*80)
    
    return df_filtered, selected_corr


if __name__ == "__main__":
    test_feature_selection()
