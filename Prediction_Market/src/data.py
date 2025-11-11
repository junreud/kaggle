"""
Data loading, preprocessing, and benchmark calculation for Hull Tactical Market Prediction.

This module provides:
- Data loading and validation
- Missing value analysis and handling
- Outlier detection and treatment
- Feature scaling strategies
- Benchmark metric calculation
"""

import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.utils import get_logger, Timer, load_config

# Initialize global logger once (will use logs/prediction_market.log)
logger = get_logger(log_file="logs/prediction_market.log", level="INFO")


class DataLoader:
    """Handle data loading, validation, and preprocessing for the competition."""
    
    def __init__(self, config_path: str = "conf/params.yaml"):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.data_dir = Path(self.config.get('paths', {}).get('data', 'data/raw/'))
        
        # Feature groups
        self.feature_groups = {
            'M': [],  # Market/Technical indicators
            'E': [],  # Economic indicators
            'I': [],  # Interest rate indicators
            'P': [],  # Price/Valuation indicators
            'V': [],  # Volatility indicators
            'S': [],  # Sentiment indicators
            'D': [],  # Dummy/Event flags
        }
        
        self.train_df = None
        self.test_df = None
        
    def load_data(
        self, 
        train_path: Optional[str] = None, 
        test_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load train and test data from CSV files.
        
        Args:
            train_path: Path to train.csv (optional)
            test_path: Path to test.csv (optional)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        with Timer("Loading data", logger):
            # Use provided paths or default from config
            train_path = train_path or self.data_dir / 'train.csv'
            test_path = test_path or self.data_dir / 'test.csv'
            
            logger.info(f"Loading train data from {train_path}")
            self.train_df = pd.read_csv(train_path)
            
            logger.info(f"Loading test data from {test_path}")
            self.test_df = pd.read_csv(test_path)
            
            logger.info(f"Train shape: {self.train_df.shape}")
            logger.info(f"Test shape: {self.test_df.shape}")
            
            # Categorize features by group
            self._categorize_features()
            
            # Sort by date_id to ensure chronological order
            self.train_df = self.train_df.sort_values('date_id').reset_index(drop=True)
            self.test_df = self.test_df.sort_values('date_id').reset_index(drop=True)
            
            logger.info("Data loaded and sorted by date_id")
            
        return self.train_df.copy(), self.test_df.copy()
    
    def _categorize_features(self) -> None:
        """Categorize features into groups (M, E, I, P, V, S, D)."""
        for col in self.train_df.columns:
            for group in self.feature_groups.keys():
                if col.startswith(group) and col[1:].isdigit():
                    self.feature_groups[group].append(col)
                    break
        
        logger.info("Feature groups identified:")
        for group, features in self.feature_groups.items():
            logger.info(f"  {group}: {len(features)} features")
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Perform comprehensive data quality checks.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dictionary with quality metrics
        """
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicates': df.duplicated().sum(),
            'duplicate_date_ids': df['date_id'].duplicated().sum(),
            'missing_summary': {},
            'date_range': (df['date_id'].min(), df['date_id'].max()),
            'date_gaps': self._check_date_gaps(df),
        }
        
        # Check for duplicates
        if quality_report['duplicates'] > 0:
            logger.warning(f"Found {quality_report['duplicates']} duplicate rows!")
        
        if quality_report['duplicate_date_ids'] > 0:
            logger.warning(f"Found {quality_report['duplicate_date_ids']} duplicate date_ids!")
        
        # Missing value analysis by feature group
        for group, features in self.feature_groups.items():
            group_features = [f for f in features if f in df.columns]
            if group_features:
                missing_pct = df[group_features].isnull().sum() / len(df) * 100
                quality_report['missing_summary'][group] = {
                    'features': len(group_features),
                    'missing_count': df[group_features].isnull().sum().sum(),
                    'avg_missing_pct': missing_pct.mean(),
                    'max_missing_pct': missing_pct.max(),
                }
        
        return quality_report
    
    def _check_date_gaps(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Check for gaps in date_id sequence.
        
        Args:
            df: DataFrame with date_id column
            
        Returns:
            List of (start, end) tuples representing gaps
        """
        date_ids = df['date_id'].sort_values().values
        gaps = []
        
        for i in range(len(date_ids) - 1):
            if date_ids[i+1] - date_ids[i] > 1:
                gaps.append((date_ids[i], date_ids[i+1]))
        
        if gaps:
            logger.warning(f"Found {len(gaps)} gaps in date_id sequence")
        
        return gaps
    
    def analyze_missing_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze missing value patterns (MCAR, MAR, MNAR).
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            DataFrame with missing pattern analysis
        """
        missing_stats = []
        
        for group, features in self.feature_groups.items():
            group_features = [f for f in features if f in df.columns]
            
            for col in group_features:
                if df[col].isnull().sum() > 0:
                    missing_pct = df[col].isnull().sum() / len(df) * 100
                    
                    # Check if missingness is random
                    # Simple test: correlation with date_id
                    is_missing = df[col].isnull().astype(int)
                    date_corr = np.corrcoef(df['date_id'], is_missing)[0, 1]
                    
                    # Pattern classification (simplified)
                    if abs(date_corr) < 0.1:
                        pattern = 'MCAR'  # Missing Completely At Random
                    elif abs(date_corr) < 0.5:
                        pattern = 'MAR'   # Missing At Random
                    else:
                        pattern = 'MNAR'  # Missing Not At Random
                    
                    missing_stats.append({
                        'feature': col,
                        'group': group,
                        'missing_count': df[col].isnull().sum(),
                        'missing_pct': missing_pct,
                        'date_correlation': date_corr,
                        'pattern': pattern,
                    })
        
        return pd.DataFrame(missing_stats).sort_values('missing_pct', ascending=False)
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        train_df: Optional[pd.DataFrame] = None,
        strategy: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values with group-specific strategies.
        
        Strategy by feature group:
        - E/I/P: Forward-fill with limit + median (economic indicators change slowly)
        - M/V/S: Rolling median to prevent leakage (market indicators are dynamic)
        - D: Fill with 0 (dummy flags)
        
        Args:
            df: DataFrame to process
            train_df: Training data for fitting (optional)
            strategy: Custom strategy per group (optional)
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        default_strategy = {
            'E': 'ffill_median',
            'I': 'ffill_median', 
            'P': 'ffill_median',
            'M': 'rolling_median',
            'V': 'rolling_median',
            'S': 'rolling_median',
            'D': 'zero',
        }
        
        strategy = strategy or default_strategy
        window = self.config.get('preprocessing', {}).get('rolling_window', 20)
        ffill_limit = self.config.get('preprocessing', {}).get('ffill_limit', 5)
        
        for group, features in self.feature_groups.items():
            group_features = [f for f in features if f in df.columns]
            
            if not group_features:
                continue
            
            group_strategy = strategy.get(group, 'median')
            
            if group_strategy == 'ffill_median':
                # Forward fill with limit, then median
                for col in group_features:
                    df[col] = df[col].fillna(method='ffill', limit=ffill_limit)
                    
                    # Use training median if available, otherwise current median
                    if train_df is not None and col in train_df.columns:
                        median_val = train_df[col].median()
                    else:
                        median_val = df[col].median()
                    
                    df[col] = df[col].fillna(median_val)
                    
            elif group_strategy == 'rolling_median':
                # Rolling median to prevent leakage
                for col in group_features:
                    # Use expanding window for early rows
                    rolling_median = df[col].expanding(min_periods=1).median()
                    df[col] = df[col].fillna(rolling_median)
                    
            elif group_strategy == 'zero':
                # Fill with 0 (for dummy variables)
                df[group_features] = df[group_features].fillna(0)
                
            elif group_strategy == 'median':
                # Simple median fill
                for col in group_features:
                    if train_df is not None and col in train_df.columns:
                        median_val = train_df[col].median()
                    else:
                        median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
        
        logger.info(f"Missing values handled. Remaining: {df.isnull().sum().sum()}")
        
        return df
    
    def detect_outliers(
        self, 
        df: pd.DataFrame, 
        method: str = 'mad',
        threshold: float = 4.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect outliers using MAD (Median Absolute Deviation) or IQR method.
        
        Args:
            df: DataFrame to analyze
            method: 'mad' or 'iqr'
            threshold: Threshold for outlier detection (MAD multiplier or IQR multiplier)
            
        Returns:
            Tuple of (outlier_mask DataFrame, outlier_summary DataFrame)
        """
        outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        outlier_summary = []
        
        # Get numeric columns (exclude date_id and target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                       'market_forward_excess_returns', 'is_scored']
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        for col in numeric_cols:
            if method == 'mad':
                # Median Absolute Deviation
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                
                if mad == 0:
                    continue
                
                modified_z_scores = 0.6745 * (df[col] - median) / mad
                outliers = np.abs(modified_z_scores) > threshold
                
            elif method == 'iqr':
                # Interquartile Range
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            outlier_mask[col] = outliers
            
            if outliers.sum() > 0:
                outlier_summary.append({
                    'feature': col,
                    'outlier_count': outliers.sum(),
                    'outlier_pct': outliers.sum() / len(df) * 100,
                    'min': df.loc[outliers, col].min() if outliers.sum() > 0 else None,
                    'max': df.loc[outliers, col].max() if outliers.sum() > 0 else None,
                })
        
        summary_df = pd.DataFrame(outlier_summary)
        if len(summary_df) > 0:
            summary_df = summary_df.sort_values('outlier_pct', ascending=False)
        
        logger.info(f"Outlier detection complete. Found {len(summary_df)} features with outliers")
        
        return outlier_mask, summary_df
    
    def winsorize_outliers(
        self, 
        df: pd.DataFrame, 
        limits: Tuple[float, float] = (0.01, 0.01)
    ) -> pd.DataFrame:
        """
        Winsorize outliers by capping at percentiles.
        
        Args:
            df: DataFrame to process
            limits: (lower, upper) percentile limits
            
        Returns:
            DataFrame with winsorized values
        """
        df = df.copy()
        
        # Get numeric columns (exclude date_id and target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                       'market_forward_excess_returns', 'is_scored']
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        for col in numeric_cols:
            if df[col].notna().sum() > 0:
                df[col] = stats.mstats.winsorize(df[col].values, limits=limits)
        
        logger.info(f"Winsorization complete with limits {limits}")
        
        return df
    
    def scale_features(
        self, 
        df: pd.DataFrame, 
        train_df: Optional[pd.DataFrame] = None,
        method: str = 'robust',
        by_group: bool = True
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Scale features using group-specific strategies.
        
        IMPORTANT: Scalers are fitted on training data only to prevent leakage.
        
        Args:
            df: DataFrame to scale
            train_df: Training data for fitting scalers
            method: 'robust' or 'standard'
            by_group: Whether to scale by feature group
            
        Returns:
            Tuple of (scaled DataFrame, scalers dict)
        """
        df = df.copy()
        scalers = {}
        
        # Determine which data to fit on
        fit_df = train_df if train_df is not None else df
        
        if method == 'robust':
            scaler_class = RobustScaler
        elif method == 'standard':
            scaler_class = StandardScaler
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if by_group:
            # Scale by feature group
            for group, features in self.feature_groups.items():
                group_features = [f for f in features if f in df.columns]
                
                if not group_features:
                    continue
                
                # Skip dummy variables
                if group == 'D':
                    continue
                
                scaler = scaler_class()
                
                # Fit on training data
                scaler.fit(fit_df[group_features].fillna(0))
                
                # Transform the data
                df[group_features] = scaler.transform(df[group_features].fillna(0))
                
                scalers[group] = scaler
                
                logger.info(f"Scaled {len(group_features)} features in group {group}")
        else:
            # Scale all numeric features together
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate', 
                           'market_forward_excess_returns', 'is_scored']
            scale_cols = [c for c in numeric_cols if c not in exclude_cols and not c.startswith('D')]
            
            scaler = scaler_class()
            scaler.fit(fit_df[scale_cols].fillna(0))
            df[scale_cols] = scaler.transform(df[scale_cols].fillna(0))
            
            scalers['all'] = scaler
            
            logger.info(f"Scaled {len(scale_cols)} features")
        
        return df, scalers
    
    def split_train_val(
        self, 
        df: pd.DataFrame, 
        val_ratio: float = 0.2,
        method: str = 'temporal'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and validation sets.
        
        Args:
            df: DataFrame to split
            val_ratio: Ratio of validation data
            method: 'temporal' (chronological) or 'random'
            
        Returns:
            Tuple of (train_df, val_df)
        """
        if method == 'temporal':
            # Temporal split (chronological)
            split_idx = int(len(df) * (1 - val_ratio))
            train_df = df.iloc[:split_idx].copy()
            val_df = df.iloc[split_idx:].copy()
            
            logger.info(f"Temporal split: Train {len(train_df)}, Val {len(val_df)}")
            logger.info(f"Train date range: {train_df['date_id'].min()} - {train_df['date_id'].max()}")
            logger.info(f"Val date range: {val_df['date_id'].min()} - {val_df['date_id'].max()}")
            
        elif method == 'random':
            # Random split (not recommended for time series)
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=42)
            
            logger.warning("Random split used - not recommended for time series data!")
        
        return train_df, val_df


def calculate_benchmark_score(
    df: pd.DataFrame,
    allocation: float = 1.0,
    return_details: bool = True
) -> Union[float, Tuple[float, pd.Series, Dict]]:
    """
    Calculate benchmark score with allocation=1.0 (market-equivalent strategy).
    
    Metric: Modified Sharpe Ratio with volatility penalty
    
    score = (mean(strategy_returns) / std(strategy_returns)) / vol_penalty
    
    where:
    - strategy_returns = allocation × forward_returns
    - vol_penalty = 1 + max(0, (strategy_vol / market_vol) - 1.2)
    
    Args:
        df: DataFrame with 'forward_returns' column
        allocation: Fixed allocation value (0 to 2)
        return_details: Whether to return detailed metrics
        
    Returns:
        If return_details=False: benchmark_score
        If return_details=True: (benchmark_score, strategy_returns, metrics_dict)
    """
    # Validate allocation
    if not 0 <= allocation <= 2:
        raise ValueError(f"Allocation must be between 0 and 2, got {allocation}")
    
    # Calculate strategy returns
    strategy_returns = allocation * df['forward_returns']
    
    # Calculate volatilities
    strategy_vol = strategy_returns.std()
    market_vol = df['forward_returns'].std()
    
    # Volatility penalty
    vol_ratio = strategy_vol / market_vol if market_vol > 0 else 1.0
    vol_penalty = 1 + max(0, vol_ratio - 1.2)
    
    # Sharpe ratio
    mean_return = strategy_returns.mean()
    sharpe = mean_return / strategy_vol if strategy_vol > 0 else 0.0
    
    # Final score
    benchmark_score = sharpe / vol_penalty
    
    if not return_details:
        return benchmark_score
    
    # Calculate additional metrics
    annual_factor = 252  # Trading days per year
    
    # Cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Maximum drawdown
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Annualized metrics
    annual_return = mean_return * annual_factor
    annual_vol = strategy_vol * np.sqrt(annual_factor)
    annual_sharpe = sharpe * np.sqrt(annual_factor)
    
    # Underperformance check
    market_mean_return = df['forward_returns'].mean()
    underperformance = mean_return < market_mean_return
    
    # Win rate
    win_rate = (strategy_returns > 0).sum() / len(strategy_returns)
    
    metrics = {
        'score': benchmark_score,
        'sharpe': sharpe,
        'annual_sharpe': annual_sharpe,
        'vol_penalty': vol_penalty,
        'vol_ratio': vol_ratio,
        'mean_return': mean_return,
        'annual_return': annual_return,
        'strategy_vol': strategy_vol,
        'annual_vol': annual_vol,
        'market_vol': market_vol,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'underperformance': underperformance,
        'allocation': allocation,
        'n_days': len(df),
    }
    
    logger.info(f"Benchmark Score: {benchmark_score:.4f}")
    logger.info(f"Annual Sharpe: {annual_sharpe:.4f}")
    logger.info(f"Volatility Penalty: {vol_penalty:.4f}")
    logger.info(f"Max Drawdown: {max_drawdown:.2%}")
    
    return benchmark_score, strategy_returns, metrics


if __name__ == "__main__":
    # Test data loading and processing
    from src.utils import set_seed
    
    set_seed(42)
    
    # Initialize loader
    loader = DataLoader()
    
    # Load data
    train_df, test_df = loader.load_data()
    
    # Check data quality
    quality = loader.check_data_quality(train_df)
    print("\n=== Data Quality Report ===")
    print(f"Total rows: {quality['total_rows']}")
    print(f"Duplicates: {quality['duplicates']}")
    print(f"Date range: {quality['date_range']}")
    
    # Analyze missing patterns
    missing_patterns = loader.analyze_missing_patterns(train_df)
    print("\n=== Top 10 Missing Features ===")
    print(missing_patterns.head(10))
    
    # Detect outliers
    outlier_mask, outlier_summary = loader.detect_outliers(train_df, method='mad', threshold=4.0)
    print("\n=== Top 10 Features with Outliers ===")
    print(outlier_summary.head(10))
    
    # Calculate benchmark
    score, returns, metrics = calculate_benchmark_score(train_df, allocation=1.0)
    print("\n=== Benchmark Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
