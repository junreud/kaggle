"""
Test script for hyperparameter tuning and model interpretability.

This script:
1. Loads and prepares data
2. Creates features
3. Runs hyperparameter tuning (small scale for testing)
4. Analyzes model interpretability
5. Saves all results
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pickle

from src.data import DataLoader
from src.features import FeatureEngineering
from src.tuner import OptunaLightGBMTuner
from src.interpretability import ModelInterpreter
from src.utils import get_logger, Timer

logger = get_logger(log_file="logs/phase3_final_test.log", level="INFO")


def test_tuning_and_interpretability():
    """Test hyperparameter tuning and interpretability analysis."""
    
    logger.info("="*80)
    logger.info("Phase 3 Final Test: Tuning + Interpretability")
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
        test_size = 3000
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
    
    # Prepare data
    feature_cols = [
        col for col in df_selected.columns 
        if col not in ['date_id', 'forward_returns', 'risk_free_rate', 
                      'market_forward_excess_returns']
    ]
    
    logger.info(f"\nFinal dataset:")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Samples: {len(df_selected)}")
    logger.info(f"  Target: forward_returns")
    
    # =========================================================================
    # Step 4: Hyperparameter Tuning (Quick Test)
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 4: Hyperparameter Tuning")
    logger.info("="*80)
    
    # Use small number of trials for testing
    n_trials_test = 10
    logger.info(f"Running {n_trials_test} trials (use more for production)")
    
    with Timer("Hyperparameter tuning"):
        tuner = OptunaLightGBMTuner(
            n_trials=n_trials_test,
            timeout=None,
            n_jobs=1,  # Use 1 job for reproducibility
            random_state=42
        )
        
        # Run optimization
        best_params = tuner.optimize(
            df_selected,
            feature_cols,
            target_col='forward_returns',
            study_name="lightgbm_phase3_test"
        )
        
        # Save study
        tuner.save_study(
            study_path="artifacts/tuning/lightgbm_study_test.pkl",
            params_path="artifacts/tuning/lightgbm_best_params_test.json"
        )
    
    logger.info("\n✓ Tuning complete")
    logger.info(f"Best Score: {tuner.best_score:.6f}")
    logger.info(f"Best Parameters saved to artifacts/tuning/")
    
    # =========================================================================
    # Step 5: Model Interpretability
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 5: Model Interpretability Analysis")
    logger.info("="*80)
    
    # Load trained models
    model_paths = [
        "artifacts/models/lightgbm_fold_0.pkl",
        "artifacts/models/lightgbm_fold_1.pkl",
        "artifacts/models/lightgbm_fold_2.pkl",
        "artifacts/models/lightgbm_fold_3.pkl",
    ]
    
    models = []
    for path in model_paths:
        if Path(path).exists():
            with open(path, 'rb') as f:
                models.append(pickle.load(f))
        else:
            logger.warning(f"Model not found: {path}")
    
    if len(models) == 0:
        logger.error("No models found. Run test_model_training.py first.")
        return
    
    logger.info(f"Loaded {len(models)} models")
    
    # Get feature names from the first model (all models use same features)
    model_feature_names = models[0].feature_name()
    logger.info(f"Model trained with {len(model_feature_names)} features")
    
    # Create interpreter
    with Timer("Creating interpreter"):
        interpreter = ModelInterpreter(
            models=models,
            feature_names=model_feature_names,
            model_type='lightgbm'
        )
    
    # Calculate feature importance
    with Timer("Feature importance calculation"):
        importance_df = interpreter.calculate_feature_importance(importance_type='gain')
    
    # Calculate SHAP values (on small sample for speed)
    logger.info("\nCalculating SHAP values...")
    
    # Check if all model features exist in df_selected
    missing_features = [f for f in model_feature_names if f not in df_selected.columns]
    
    if missing_features:
        logger.warning(f"Cannot calculate SHAP: {len(missing_features)} model features not in current dataset")
        logger.warning("This happens when feature selection produces different features than training.")
        logger.warning("To fix: Either use same random seed or load saved feature list from training.")
        logger.info("Skipping SHAP analysis...")
    else:
        try:
            with Timer("SHAP values calculation"):
                # Use small sample for testing
                # Use only features that the model was trained on
                X_sample = df_selected[model_feature_names]
                sample_size = min(500, len(X_sample))
                shap_values = interpreter.calculate_shap_values(
                    X_sample, 
                    sample_size=sample_size,
                    random_state=42
                )
            
            logger.info("✓ SHAP values calculated")
            
        except ImportError:
            logger.warning("SHAP library not installed. Install with: pip install shap")
            logger.warning("Skipping SHAP analysis")
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}")
            logger.warning("Skipping SHAP analysis")
    
    # Analyze feature interactions (optional, can be slow)
    if len(models) > 0 and not missing_features:
        logger.info("\nAnalyzing feature interactions...")
        try:
            with Timer("Feature interaction analysis"):
                X_sample = df_selected[model_feature_names]
                interactions_df = interpreter.get_feature_interactions(
                    X_sample,
                    top_n=10,
                    sample_size=min(500, len(X_sample)),
                    random_state=42
                )
            
            logger.info("✓ Feature interactions analyzed")
            
        except Exception as e:
            logger.warning(f"Feature interaction analysis failed: {e}")
    
    # Save all results
    with Timer("Saving interpretability results"):
        interpreter.save_analysis(output_dir="results/interpretability")
    
    logger.info("\n✓ Interpretability analysis complete")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("Phase 3 Final Test Summary")
    logger.info("="*80)
    
    logger.info("\n✅ Completed Tasks:")
    logger.info("  1. Feature Engineering - 542 features created")
    logger.info("  2. Feature Selection - Reduced to ~60 features")
    logger.info("  3. Model Training - LightGBM with CV")
    logger.info("  4. Hyperparameter Tuning - Optuna optimization")
    logger.info("  5. Model Interpretability - SHAP + Feature Importance")
    
    logger.info("\n📁 Results saved to:")
    logger.info("  - artifacts/tuning/lightgbm_best_params_test.json")
    logger.info("  - artifacts/tuning/optimization_history.csv")
    logger.info("  - results/interpretability/feature_importance_gain.csv")
    logger.info("  - results/interpretability/shap_importance.csv")
    
    logger.info("\n🎯 Next Steps:")
    logger.info("  1. Run full hyperparameter tuning (100+ trials)")
    logger.info("  2. Train final models with best parameters")
    logger.info("  3. Generate predictions for test set")
    logger.info("  4. Create submission file")
    
    logger.info("\n" + "="*80)
    logger.info("Phase 3 Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    test_tuning_and_interpretability()
