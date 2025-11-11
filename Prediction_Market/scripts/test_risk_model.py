"""
Test script for risk prediction model.

This script:
1. Loads and prepares data
2. Creates risk labels (future volatility)
3. Trains risk forecasting model
4. Generates OOF risk predictions
5. Assesses calibration
6. Saves results
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
from src.features import FeatureEngineering
from src.risk import RiskLabeler, RiskForecaster, RiskCalibrator
from src.utils import get_logger, Timer

logger = get_logger(log_file="logs/phase4_risk_test.log", level="INFO")


def test_risk_prediction():
    """Test risk prediction pipeline."""
    
    logger.info("="*80)
    logger.info("Phase 4 Test: Risk Prediction Model")
    logger.info("="*80)
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 1: Loading Data")
    
    with Timer("Loading data"):
        data_loader = DataLoader()
        df, _ = data_loader.load_data()
        
        # Use subset for testing (to make it faster)
        test_size = 5000
        logger.info(f"Using {test_size} samples for testing")
        df = df.head(test_size)
        
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Date range: {df['date_id'].min()} to {df['date_id'].max()}")
    
    # =========================================================================
    # Step 2: Feature Engineering
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 2: Feature Engineering")
    
    with Timer("Feature engineering"):
        fe = FeatureEngineering()
        df_features = fe.fit_transform(df)
        
        logger.info(f"Features created: {len(df_features.columns)}")
    
    # =========================================================================
    # Step 3: Feature Selection
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 3: Feature Selection")
    
    with Timer("Feature selection"):
        # Select top features
        df_selected, selected_features = fe.select_features_by_importance(
            df_features,
            target_col='forward_returns',
            method='correlation',
            top_n=100
        )
        
        # Remove correlated features
        df_selected, removed_features = fe.remove_correlated_features(df_selected, threshold=0.95)
        
        logger.info(f"Selected features: {len(df_selected.columns) - 4}")  # Exclude meta columns
    
    # Prepare feature columns
    feature_cols = [
        col for col in df_selected.columns 
        if col not in ['date_id', 'forward_returns', 'risk_free_rate', 
                      'market_forward_excess_returns']
    ]
    
    logger.info(f"\nDataset:")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Samples: {len(df_selected)}")
    
    # =========================================================================
    # Step 4: Create Risk Labels
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 4: Creating Risk Labels")
    logger.info("="*80)
    
    with Timer("Risk label creation"):
        risk_labeler = RiskLabeler()
        df_selected = risk_labeler.fit_transform(df_selected, target_col='forward_returns')
        
        logger.info(f"Risk labels created: {df_selected['risk_label'].notna().sum()} valid samples")
    
    # =========================================================================
    # Step 5: Train Risk Forecaster
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 5: Training Risk Forecaster")
    logger.info("="*80)
    
    with Timer("Risk model training"):
        risk_forecaster = RiskForecaster()
        
        # Train with CV
        oof_sigma_hat, models = risk_forecaster.train(
            df_selected,
            feature_cols,
            risk_col='risk_label',
            n_folds=5,
            early_stopping_rounds=50
        )
        
        logger.info(f"\n✓ Training complete")
        logger.info(f"  Models trained: {len(models)}")
        logger.info(f"  OOF predictions: {np.sum(~np.isnan(oof_sigma_hat))}")
    
    # =========================================================================
    # Step 6: Save OOF Predictions
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 6: Saving OOF Predictions")
    
    with Timer("Saving OOF predictions"):
        # Add OOF predictions to dataframe
        df_selected['sigma_hat'] = oof_sigma_hat
        
        # Save to parquet
        output_path = Path("artifacts/oof_sigma_hat.parquet")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_selected[['date_id', 'risk_label', 'sigma_hat']].to_parquet(
            output_path,
            index=False
        )
        
        logger.info(f"✓ OOF predictions saved to {output_path}")
    
    # =========================================================================
    # Step 7: Save Models
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 7: Saving Models")
    
    with Timer("Saving models"):
        risk_forecaster.save_models(output_dir="artifacts/models")
        
        logger.info("✓ Models saved")
    
    # =========================================================================
    # Step 8: Feature Importance
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 8: Feature Importance Analysis")
    
    with Timer("Feature importance"):
        importance_df = risk_forecaster.get_feature_importance(importance_type='gain')
        
        logger.info(f"\nTop 20 Features for Risk Prediction:")
        for idx, row in importance_df.head(20).iterrows():
            logger.info(f"  {idx+1:2d}. {row['feature']:30s}: {row['importance']:.4f} (±{row['std']:.4f})")
        
        # Save feature importance
        importance_path = Path("results/risk_feature_importance.csv")
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(importance_path, index=False)
        
        logger.info(f"\n✓ Feature importance saved to {importance_path}")
    
    # =========================================================================
    # Step 9: Calibration Assessment
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 9: Calibration Assessment")
    logger.info("="*80)
    
    with Timer("Calibration assessment"):
        calibrator = RiskCalibrator()
        
        # Get valid samples
        valid_idx = ~(np.isnan(df_selected['risk_label']) | np.isnan(oof_sigma_hat))
        
        y_true = df_selected.loc[valid_idx, 'risk_label'].values
        y_pred = oof_sigma_hat[valid_idx]
        
        # Assess calibration
        calib_results = calibrator.assess_calibration(y_true, y_pred)
        
        # Save calibration report
        calibrator.save_calibration_report(
            calib_results,
            output_path="results/risk_calibration_report.csv"
        )
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("Phase 4 Test Summary")
    logger.info("="*80)
    
    logger.info("\n✅ Completed Tasks:")
    logger.info("  1. Risk Label Creation - Future volatility calculation")
    logger.info("  2. Feature Engineering - Reused from Phase 3")
    logger.info("  3. Risk Model Training - LightGBM with CV")
    logger.info("  4. OOF Predictions - sigma_hat generated")
    logger.info("  5. Calibration Assessment - Quality metrics calculated")
    
    logger.info("\n📁 Results saved to:")
    logger.info("  - artifacts/oof_sigma_hat.parquet")
    logger.info("  - artifacts/models/risk_lgbm_fold_*.pkl")
    logger.info("  - results/risk_feature_importance.csv")
    logger.info("  - results/risk_calibration_report.csv")
    
    # Print key metrics
    logger.info("\n📊 Key Metrics:")
    logger.info(f"  Calibration Error: {calib_results['calibration_error']:.6f}")
    logger.info(f"  Correlation: {calib_results['correlation']:.4f}")
    logger.info(f"  Valid predictions: {valid_idx.sum()}/{len(df_selected)} ({valid_idx.sum()/len(df_selected)*100:.1f}%)")
    
    logger.info("\n🎯 Next Steps:")
    logger.info("  1. Run full training on entire dataset")
    logger.info("  2. Tune hyperparameters with Optuna")
    logger.info("  3. Implement ensemble risk models")
    logger.info("  4. Move to Phase 5: Position mapping & risk control")
    
    logger.info("\n" + "="*80)
    logger.info("Phase 4 Test Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    test_risk_prediction()
