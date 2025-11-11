"""
Test feature engineering pipeline.

This script demonstrates:
1. Loading data
2. Creating engineered features
3. Validating feature creation
4. Saving feature statistics
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
logger = get_logger(log_file="logs/features_test.log", level="INFO")


def test_feature_engineering():
    """Test feature engineering pipeline."""
    logger.info("="*80)
    logger.info("Testing Feature Engineering")
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
    
    logger.info(f"\nFinal data shape: {train_features.shape}")
    logger.info(f"Number of features: {len(train_features.columns) - 2}")  # Exclude date_id and forward_returns
    
    # Check for NaN values
    logger.info("\n" + "="*80)
    logger.info("Feature Quality Check")
    logger.info("="*80)
    
    nan_counts = train_features.isnull().sum()
    nan_features = nan_counts[nan_counts > 0].sort_values(ascending=False)
    
    if len(nan_features) > 0:
        logger.info(f"\nFeatures with NaN values: {len(nan_features)}")
        logger.info("\nTop 10 features with most NaN:")
        for feat, count in nan_features.head(10).items():
            pct = (count / len(train_features)) * 100
            logger.info(f"  {feat}: {count} ({pct:.2f}%)")
    else:
        logger.info("\n✓ No NaN values found!")
    
    # Feature statistics
    logger.info("\n" + "="*80)
    logger.info("Feature Statistics")
    logger.info("="*80)
    
    # Original features
    logger.info(f"\nOriginal features: {len(feature_eng.original_features)}")
    
    # Engineered features by type
    engineered = feature_eng.engineered_features
    
    rolling_features = [f for f in engineered if '_roll_' in f or '_dev_' in f or '_zscore_' in f]
    lag_features = [f for f in engineered if '_lag_' in f]
    diff_features = [f for f in engineered if '_diff_' in f or '_pct_' in f or '_accel' in f]
    interaction_features = [f for f in engineered if '_x_' in f or '_div_' in f]
    technical_features = [f for f in engineered if any(x in f for x in ['_rsi', '_momentum_', '_bb_'])]
    regime_features = [f for f in engineered if '_vol' in f and ('high' in f or 'low' in f)]
    
    logger.info(f"\nEngineered features by type:")
    logger.info(f"  Rolling features: {len(rolling_features)}")
    logger.info(f"  Lag features: {len(lag_features)}")
    logger.info(f"  Difference features: {len(diff_features)}")
    logger.info(f"  Interaction features: {len(interaction_features)}")
    logger.info(f"  Technical features: {len(technical_features)}")
    logger.info(f"  Regime features: {len(regime_features)}")
    logger.info(f"  Total engineered: {len(engineered)}")
    
    # Save sample features
    logger.info("\n" + "="*80)
    logger.info("Saving Sample Features")
    logger.info("="*80)
    
    output_dir = Path("results/features")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save first few rows
    sample_df = train_features.head(100)
    sample_path = output_dir / "feature_sample.csv"
    sample_df.to_csv(sample_path, index=False)
    logger.info(f"\n✓ Sample features saved to {sample_path}")
    
    # Save feature names
    feature_names = {
        'original': feature_eng.original_features,
        'rolling': rolling_features[:10],  # Sample
        'lag': lag_features[:10],
        'difference': diff_features[:10],
        'interaction': interaction_features[:10],
        'technical': technical_features[:10],
        'regime': regime_features
    }
    
    feature_list_path = output_dir / "feature_names.txt"
    with open(feature_list_path, 'w') as f:
        for feat_type, feat_list in feature_names.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"{feat_type.upper()} FEATURES ({len(feat_list)})\n")
            f.write(f"{'='*60}\n")
            for feat in feat_list:
                f.write(f"{feat}\n")
    
    logger.info(f"✓ Feature names saved to {feature_list_path}")
    
    # Feature correlation analysis
    logger.info("\n" + "="*80)
    logger.info("Feature Correlation Analysis")
    logger.info("="*80)
    
    if 'forward_returns' in train_features.columns:
        # Calculate correlation with target
        correlations = train_features.corr()['forward_returns'].abs().sort_values(ascending=False)
        
        # Remove target itself
        correlations = correlations[correlations.index != 'forward_returns']
        
        logger.info(f"\nTop 20 features correlated with forward_returns:")
        for feat, corr in correlations.head(20).items():
            logger.info(f"  {feat}: {corr:.4f}")
        
        # Save correlations
        corr_path = output_dir / "feature_correlations.csv"
        correlations.to_csv(corr_path, header=['correlation'])
        logger.info(f"\n✓ Correlations saved to {corr_path}")
    
    logger.info("\n" + "="*80)
    logger.info("Feature Engineering Test Complete")
    logger.info("="*80)
    
    return train_features


if __name__ == "__main__":
    test_feature_engineering()
