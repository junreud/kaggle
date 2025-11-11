"""
Test model training with LightGBM.

This script demonstrates:
1. Loading and preparing data
2. Feature engineering
3. Feature selection
4. Model training with CV
5. OOF predictions
6. Feature importance analysis
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
from src.models import ReturnPredictor, create_return_predictor
from src.utils import get_logger, Timer, set_seed

# Initialize logger
logger = get_logger(log_file="logs/model_training_test.log", level="INFO")


def test_model_training():
    """Test model training pipeline."""
    logger.info("="*80)
    logger.info("Testing Model Training Pipeline")
    logger.info("="*80)
    
    # Set seed
    set_seed(42)
    
    # Step 1: Load data
    logger.info("\n" + "="*80)
    logger.info("Step 1: Loading Data")
    logger.info("="*80)
    
    with Timer("Loading data", logger=logger):
        config_path = "conf/params.yaml"
        data_loader = DataLoader(config_path)
        train_data, _ = data_loader.load_data()
        
        logger.info(f"Train data shape: {train_data.shape}")
    
    # Step 2: Feature Engineering
    logger.info("\n" + "="*80)
    logger.info("Step 2: Feature Engineering")
    logger.info("="*80)
    
    feature_eng = create_feature_engineering(config_path)
    
    with Timer("Feature engineering", logger=logger):
        train_features = feature_eng.fit_transform(train_data, date_col='date_id')
    
    logger.info(f"Features created: {train_features.shape[1] - 2}")
    
    # Step 3: Feature Selection
    logger.info("\n" + "="*80)
    logger.info("Step 3: Feature Selection")
    logger.info("="*80)
    
    with Timer("Feature selection", logger=logger):
        # Select top 100 features by correlation
        train_selected, selected_features = feature_eng.select_features_by_importance(
            train_features,
            target_col='forward_returns',
            method='correlation',
            top_n=100
        )
        
        # Remove highly correlated features
        train_final, removed_features = feature_eng.remove_correlated_features(
            train_selected,
            threshold=0.95,
            target_col='forward_returns'
        )
    
    logger.info(f"Final features: {train_final.shape[1] - 2}")
    
    # Step 4: Model Training (use small sample for faster testing)
    logger.info("\n" + "="*80)
    logger.info("Step 4: Model Training (LightGBM)")
    logger.info("="*80)
    
    # Use subset for faster testing
    sample_size = min(5000, len(train_final))
    train_sample = train_final.tail(sample_size).copy()  # Use recent data
    
    logger.info(f"Using {sample_size} samples for training")
    
    # Create model
    model = create_return_predictor(
        model_type='lightgbm',
        config_path=config_path
    )
    
    # Train with CV
    with Timer("Model training", logger=logger):
        results = model.train_cv(
            train_sample,
            target_col='forward_returns',
            date_col='date_id'
        )
    
    # Step 5: Analyze Results
    logger.info("\n" + "="*80)
    logger.info("Step 5: Results Analysis")
    logger.info("="*80)
    
    logger.info(f"\nTraining Results:")
    logger.info(f"  Mean CV Score (RMSE): {results['mean_score']:.6f}")
    logger.info(f"  Std CV Score: {results['std_score']:.6f}")
    logger.info(f"  OOF Score: {results['oof_score']:.6f}")
    logger.info(f"  Number of features: {results['n_features']}")
    logger.info(f"  Number of samples: {results['n_samples']}")
    
    # Fold scores
    logger.info(f"\nFold Scores:")
    for i, score in enumerate(results['fold_scores']):
        logger.info(f"  Fold {i+1}: {score:.6f}")
    
    # Step 6: Save Results
    logger.info("\n" + "="*80)
    logger.info("Step 6: Saving Results")
    logger.info("="*80)
    
    output_dir = Path("results/model_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    model.save_models(output_dir="artifacts/models")
    
    # Save OOF predictions
    model.save_oof_predictions(output_path="artifacts/oof_r_hat.csv")
    
    # Save training summary
    summary = pd.DataFrame({
        'Metric': [
            'Mean CV Score (RMSE)',
            'Std CV Score',
            'OOF Score',
            'Number of Features',
            'Number of Samples',
            'Number of Folds'
        ],
        'Value': [
            results['mean_score'],
            results['std_score'],
            results['oof_score'],
            results['n_features'],
            results['n_samples'],
            len(results['fold_scores'])
        ]
    })
    
    summary_path = output_dir / "training_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"✓ Training summary saved to {summary_path}")
    
    # Save feature importance
    importance_path = output_dir / "feature_importance.csv"
    results['feature_importance'].to_csv(importance_path, index=False)
    logger.info(f"✓ Feature importance saved to {importance_path}")
    
    # Analyze OOF predictions
    logger.info("\n" + "="*80)
    logger.info("OOF Prediction Analysis")
    logger.info("="*80)
    
    oof_preds = results['oof_predictions']
    oof_mask = results['oof_indices']
    actual = train_sample['forward_returns'].values
    
    # Calculate correlation
    correlation = np.corrcoef(actual[oof_mask], oof_preds[oof_mask])[0, 1]
    logger.info(f"\nOOF Correlation with actual: {correlation:.4f}")
    
    # Distribution statistics
    logger.info(f"\nOOF Predictions Distribution:")
    logger.info(f"  Mean: {oof_preds[oof_mask].mean():.6f}")
    logger.info(f"  Std: {oof_preds[oof_mask].std():.6f}")
    logger.info(f"  Min: {oof_preds[oof_mask].min():.6f}")
    logger.info(f"  Max: {oof_preds[oof_mask].max():.6f}")
    
    logger.info(f"\nActual Returns Distribution:")
    logger.info(f"  Mean: {actual[oof_mask].mean():.6f}")
    logger.info(f"  Std: {actual[oof_mask].std():.6f}")
    logger.info(f"  Min: {actual[oof_mask].min():.6f}")
    logger.info(f"  Max: {actual[oof_mask].max():.6f}")
    
    logger.info("\n" + "="*80)
    logger.info("Model Training Test Complete")
    logger.info("="*80)
    
    return model, results


if __name__ == "__main__":
    test_model_training()
