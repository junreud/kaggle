"""
Cross-validation strategies for time series prediction.

This module provides:
- Time-aware cross-validation with purge and embargo
- Walk-forward validation
- Evaluation metrics computation
"""

from typing import Tuple, List, Iterator, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from src.utils import get_logger, load_config

logger = get_logger(log_file="logs/prediction_market.log", level="INFO")


class PurgedWalkForwardCV(BaseCrossValidator):
    """
    Time series cross-validator with purge and embargo.
    
    This implements a walk-forward validation strategy that:
    1. Respects temporal ordering (no data leakage)
    2. Purges overlapping samples between train and validation
    3. Applies embargo period after validation to prevent look-ahead bias
    
    Parameters
    ----------
    n_splits : int
        Number of folds
    embargo : int
        Number of periods to embargo after validation set
    purge : bool
        Whether to purge overlapping periods
    train_ratio : float
        Ratio of training data per fold (remaining goes to validation)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        embargo: int = 5,
        purge: bool = True,
        train_ratio: float = 0.8
    ):
        self.n_splits = n_splits
        self.embargo = embargo
        self.purge = purge
        self.train_ratio = train_ratio
        
    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data with date_id index or column
        y : pd.Series, optional
            Target variable
        groups : pd.Series, optional
            Group labels (date_id) for the samples
            
        Yields
        ------
        train : np.ndarray
            Training set indices
        test : np.ndarray
            Testing set indices
        """
        # Get date_id information
        if 'date_id' in X.columns:
            dates = X['date_id'].values
        elif isinstance(X.index, pd.Index) and X.index.name == 'date_id':
            dates = X.index.values
        elif groups is not None:
            dates = groups.values
        else:
            raise ValueError("date_id must be in columns, index, or provided as groups")
        
        # Get unique dates sorted
        unique_dates = np.sort(np.unique(dates))
        n_dates = len(unique_dates)
        
        # Calculate fold size
        fold_size = n_dates // self.n_splits
        
        if fold_size < 2:
            raise ValueError(
                f"n_splits={self.n_splits} is too large for {n_dates} unique dates. "
                f"Reduce n_splits or use more data."
            )
        
        logger.info(f"Starting Purged Walk-Forward CV with {self.n_splits} splits")
        logger.info(f"Total dates: {n_dates}, Fold size: {fold_size}")
        logger.info(f"Embargo: {self.embargo}, Purge: {self.purge}, Train ratio: {self.train_ratio}")
        
        for fold_idx in range(self.n_splits):
            # Define validation period for this fold
            val_start_idx = fold_idx * fold_size
            val_end_idx = min((fold_idx + 1) * fold_size, n_dates)
            
            # Skip if not enough data
            if val_start_idx >= n_dates - 1:
                break
                
            # Calculate training period
            train_start_idx = 0
            # For walk-forward CV, use all data before validation period
            train_end_idx = val_start_idx
            
            # Ensure we have enough training data
            if train_end_idx <= train_start_idx:
                logger.warning(f"Fold {fold_idx + 1}: Not enough training data, skipping")
                continue
            
            # Get date ranges
            train_dates = unique_dates[train_start_idx:train_end_idx]
            val_dates = unique_dates[val_start_idx:val_end_idx]
            
            # Apply embargo: remove dates immediately after validation
            if self.embargo > 0:
                embargo_end_idx = min(val_end_idx + self.embargo, n_dates)
                embargo_dates = unique_dates[val_end_idx:embargo_end_idx]
            else:
                embargo_dates = np.array([])
            
            # Apply purge if enabled
            if self.purge and len(embargo_dates) > 0:
                # Remove embargo dates from training
                train_dates = train_dates[~np.isin(train_dates, embargo_dates)]
            
            # Convert dates to indices
            train_idx = np.where(np.isin(dates, train_dates))[0]
            val_idx = np.where(np.isin(dates, val_dates))[0]
            
            # Log fold information
            logger.info(
                f"Fold {fold_idx + 1}/{self.n_splits}: "
                f"Train dates: {len(train_dates)} ({train_dates[0]} to {train_dates[-1]}), "
                f"Val dates: {len(val_dates)} ({val_dates[0]} to {val_dates[-1]}), "
                f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}"
            )
            
            yield train_idx, val_idx
    
    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return self.n_splits


class CVStrategy:
    """
    Cross-validation strategy manager.
    
    This class handles:
    - CV configuration from params.yaml
    - Fold generation
    - Metric aggregation across folds
    """
    
    def __init__(self, config_path: str = "conf/params.yaml"):
        """
        Initialize CV strategy with configuration.
        
        Parameters
        ----------
        config_path : str
            Path to configuration file
        """
        self.config = load_config(config_path)
        self.cv_config = self.config.get('cv', {})
        
        # CV parameters
        self.n_splits = self.cv_config.get('n_splits', 5)
        self.embargo = self.cv_config.get('embargo', 5)
        self.purge = self.cv_config.get('purge', True)
        self.train_ratio = self.cv_config.get('train_ratio', 0.8)
        
        # Initialize CV splitter
        self.cv_splitter = PurgedWalkForwardCV(
            n_splits=self.n_splits,
            embargo=self.embargo,
            purge=self.purge,
            train_ratio=self.train_ratio
        )
        
        logger.info("CV Strategy initialized")
        logger.info(f"Configuration: {self.cv_config}")
    
    def get_folds(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate CV folds.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with date_id
        y : pd.Series, optional
            Target variable
            
        Yields
        ------
        train_idx : np.ndarray
            Training indices
        val_idx : np.ndarray
            Validation indices
        """
        return self.cv_splitter.split(X, y)
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> dict:
        """
        Calculate evaluation metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True target values
        y_pred : np.ndarray
            Predicted values
        prefix : str
            Prefix for metric names
            
        Returns
        -------
        dict
            Dictionary of metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            logger.warning("No valid predictions to evaluate")
            return {}
        
        metrics = {
            f"{prefix}rmse": np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
            f"{prefix}mae": mean_absolute_error(y_true_clean, y_pred_clean),
            f"{prefix}n_samples": len(y_true_clean)
        }
        
        # Calculate correlation if enough variance
        if np.std(y_true_clean) > 1e-10 and np.std(y_pred_clean) > 1e-10:
            metrics[f"{prefix}corr"] = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
        else:
            metrics[f"{prefix}corr"] = 0.0
        
        return metrics
    
    def aggregate_fold_metrics(
        self,
        fold_metrics: List[dict]
    ) -> dict:
        """
        Aggregate metrics across folds.
        
        Parameters
        ----------
        fold_metrics : List[dict]
            List of metric dictionaries from each fold
            
        Returns
        -------
        dict
            Aggregated metrics with mean and std
        """
        if not fold_metrics:
            return {}
        
        # Get all metric names
        metric_names = set()
        for metrics in fold_metrics:
            metric_names.update(metrics.keys())
        
        aggregated = {}
        for name in metric_names:
            values = [m[name] for m in fold_metrics if name in m]
            if values:
                aggregated[f"{name}_mean"] = np.mean(values)
                aggregated[f"{name}_std"] = np.std(values)
                aggregated[f"{name}_min"] = np.min(values)
                aggregated[f"{name}_max"] = np.max(values)
        
        return aggregated


def create_cv_strategy(config_path: str = "conf/params.yaml") -> CVStrategy:
    """
    Factory function to create CV strategy.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    CVStrategy
        Configured CV strategy instance
    """
    return CVStrategy(config_path)
