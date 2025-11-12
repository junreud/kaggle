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

from utils import get_logger, Timer, load_config

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
    
    def create_missing_masks(
        self,
        df: pd.DataFrame,
        suffix: str = '_is_missing'
    ) -> pd.DataFrame:
        """
        Create binary missing indicators and gap duration features.
        
        For each feature with missing values, creates:
        1. {feature}_is_missing: Binary indicator (1 if missing, 0 otherwise)
        2. {feature}_missing_days: Days since last valid observation
        
        Args:
            df: DataFrame to process (must be sorted by date_id)
            suffix: Suffix for binary mask columns
            
        Returns:
            DataFrame with additional mask features
            
        Example:
            >>> df = loader.create_missing_masks(df)
            >>> # Creates: E1_is_missing, E1_missing_days, etc.
        """
        df = df.copy()
        mask_features_created = 0
        
        for group, features in self.feature_groups.items():
            if group == 'D':  # Skip dummy variables
                continue
                
            for col in features:
                if col not in df.columns:
                    continue
                
                # Skip if no missing values
                if not df[col].isna().any():
                    continue
                
                # 1. Binary missing indicator
                mask_col = f"{col}{suffix}"
                df[mask_col] = df[col].isna().astype(int)
                
                # 2. Missing duration (days since last valid value)
                duration_col = f"{col}_missing_days"
                is_missing = df[col].isna()
                
                # Calculate cumulative days missing
                missing_counter = 0
                duration_values = []
                
                for missing in is_missing:
                    if missing:
                        missing_counter += 1
                    else:
                        missing_counter = 0
                    duration_values.append(missing_counter)
                
                df[duration_col] = duration_values
                mask_features_created += 2
        
        logger.info(f"Created {mask_features_created} missing mask features")
        
        return df
    
    def align_announcement_dates(
        self,
        df: pd.DataFrame,
        announcement_calendar: Optional[pd.DataFrame] = None,
        default_lag: int = 15
    ) -> pd.DataFrame:
        """
        Align economic indicators with their announcement dates to prevent future leakage.
        
        CRITICAL: Economic indicators (E group) are often announced with a delay.
        For example, "February CPI" is announced on March 15th.
        Using February CPI on February 28th would be future information leakage!
        
        This method shifts values forward by announcement lag to reflect real availability.
        
        Args:
            df: DataFrame to process (must be sorted by date_id)
            announcement_calendar: Optional DataFrame with columns:
                - feature: Feature name (e.g., 'E1')
                - announcement_lag: Number of days after period end (default: 15)
            default_lag: Default announcement lag in days for features not in calendar
            
        Returns:
            DataFrame with properly aligned values
            
        Example:
            >>> # Without alignment: E1 value at date_id=100 uses data from date_id=100
            >>> # With alignment: E1 value at date_id=100 uses data from date_id=85
            >>> df_aligned = loader.align_announcement_dates(df, default_lag=15)
        """
        df = df.copy()
        
        # Simple version: Apply default lag to all E group features
        # (Full version would use announcement_calendar for feature-specific lags)
        
        if announcement_calendar is not None:
            # Use custom announcement schedule
            logger.info("Using custom announcement calendar")
            for _, row in announcement_calendar.iterrows():
                feature = row['feature']
                lag = row.get('announcement_lag', default_lag)
                
                if feature in df.columns:
                    # Shift values forward by lag days
                    df[feature] = df[feature].shift(lag)
                    logger.debug(f"Shifted {feature} by {lag} days")
        else:
            # Apply default lag to E group (economic indicators)
            e_features = [f for f in self.feature_groups.get('E', []) if f in df.columns]
            
            if e_features:
                logger.info(f"Applying {default_lag}-day announcement lag to {len(e_features)} E-group features")
                for feature in e_features:
                    df[feature] = df[feature].shift(default_lag)
        
        logger.info("Announcement date alignment completed")
        
        return df
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        train_df: Optional[pd.DataFrame] = None,
        strategy: Optional[Dict[str, str]] = None,
        max_gap: int = 10
    ) -> pd.DataFrame:
        """
        Handle missing values with TIME-SERIES AWARE strategies.
        
        IMPORTANT: Uses forward-fill only strategies to prevent future information leakage.
        
        Updated strategies (Phase 1):
        - E: LOCF only (economic indicators - prevent future leakage from interpolation)
        - I: LOCF + median fallback (interest rates)
        - P: LOCF only (price/valuation - prevent future leakage)
        - M: EWMA (market indicators - recent data weighted more)
        - V: EWMA (volatility indicators)
        - S: EWMA (sentiment indicators)
        - D: Fill with 0 (dummy flags)
        
        Args:
            df: DataFrame to process (must be sorted by date_id)
            train_df: Training data for fitting fallback values (optional)
            strategy: Custom strategy per group (optional)
            max_gap: Maximum number of periods to forward-fill (default: 10)
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        # Updated default strategy - NO interpolate for E/I/P to prevent future leakage
        default_strategy = {
            'E': 'locf',        # ✅ LOCF only (was: interpolate)
            'I': 'locf_median', # ✅ LOCF + median fallback
            'P': 'locf',        # ✅ LOCF only (was: interpolate)
            'M': 'ewma',        # EWMA for market indicators
            'V': 'ewma',        # EWMA for volatility
            'S': 'ewma',        # EWMA for sentiment
            'D': 'zero',        # Zero fill for dummy variables
        }
        
        strategy = strategy or default_strategy
        ewma_span = self.config.get('preprocessing', {}).get('ewma_span', 10)
        
        for group, features in self.feature_groups.items():
            group_features = [f for f in features if f in df.columns]
            
            if not group_features:
                continue
            
            group_strategy = strategy.get(group, 'locf')
            
            if group_strategy == 'locf':
                # Last Observation Carried Forward with max gap limit
                for col in group_features:
                    # LOCF with limit to prevent excessive forward-filling
                    df[col] = df[col].ffill(limit=max_gap)
                    
                    # Fallback: training median (no future information)
                    if df[col].isna().any():
                        if train_df is not None and col in train_df.columns:
                            fallback_val = train_df[col].median()
                        else:
                            fallback_val = df[col].median()
                        df[col] = df[col].fillna(fallback_val)
                        
            elif group_strategy == 'locf_median':
                # LOCF with median fallback (for interest rates)
                for col in group_features:
                    df[col] = df[col].ffill(limit=max_gap)
                    
                    if df[col].isna().any():
                        if train_df is not None and col in train_df.columns:
                            fallback_val = train_df[col].median()
                        else:
                            fallback_val = df[col].median()
                        df[col] = df[col].fillna(fallback_val)
                    
            elif group_strategy == 'interpolate':
                # Legacy: Time-series linear interpolation (DEPRECATED for E/I/P)
                logger.warning(f"Group {group}: 'interpolate' strategy may cause future leakage. Consider 'locf' instead.")
                for col in group_features:
                    # Linear interpolation (respects time order)
                    df[col] = df[col].interpolate(
                        method='linear',
                        limit_direction='forward',
                        limit=max_gap
                    )
                    
                    # Fill remaining with training median or current median
                    if df[col].isna().any():
                        if train_df is not None and col in train_df.columns:
                            fallback_val = train_df[col].median()
                        else:
                            fallback_val = df[col].median()
                        df[col] = df[col].fillna(fallback_val)
                    
            elif group_strategy == 'ewma':
                # Exponentially weighted moving average interpolation
                for col in group_features:
                    mask = df[col].isna()
                    if mask.any():
                        # Calculate EWMA (gives more weight to recent values)
                        ewma = df[col].ewm(span=ewma_span, min_periods=1).mean()
                        df.loc[mask, col] = ewma[mask]
                        
                        # Fill any remaining NaNs at the beginning
                        if df[col].isna().any():
                            df[col] = df[col].bfill(limit=max_gap)
                            
                            # Final fallback
                            if df[col].isna().any():
                                if train_df is not None and col in train_df.columns:
                                    fallback_val = train_df[col].median()
                                else:
                                    fallback_val = df[col].median()
                                df[col] = df[col].fillna(fallback_val)
                    
            elif group_strategy == 'zero':
                # Fill with 0 (for dummy variables)
                df[group_features] = df[group_features].fillna(0)
        
        logger.info(f"Missing values handled (LOCF-based, no future leakage). Remaining: {df.isnull().sum().sum()}")
        
        return df
    
    def detect_outliers(
        self, 
        df: pd.DataFrame, 
        method: str = 'rolling_mad',
        threshold: float = 4.0,
        window: int = 60
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect outliers using TIME-SERIES AWARE methods (rolling window based).
        
        Improved methods for time series:
        - 'rolling_mad': Rolling Median Absolute Deviation (local baseline)
        - 'rolling_iqr': Rolling IQR (local quantiles)
        - 'ewma': EWMA-based detection (recent data weighted more)
        - 'mad': Global MAD (legacy method)
        - 'iqr': Global IQR (legacy method)
        
        Args:
            df: DataFrame to analyze
            method: Detection method
            threshold: Threshold multiplier
            window: Rolling window size (for rolling methods)
            
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
            if method == 'rolling_mad':
                # Rolling Median Absolute Deviation (time-series aware)
                rolling_median = df[col].rolling(window=window, min_periods=10, center=False).median()
                rolling_mad = df[col].rolling(window=window, min_periods=10, center=False).apply(
                    lambda x: np.median(np.abs(x - np.median(x))), raw=True
                )
                
                # Avoid division by zero
                rolling_mad = rolling_mad.replace(0, np.nan)
                
                if rolling_mad.isna().all():
                    continue
                
                # Calculate modified z-scores based on rolling window
                modified_z_scores = 0.6745 * (df[col] - rolling_median) / rolling_mad
                outliers = np.abs(modified_z_scores) > threshold
                
            elif method == 'rolling_iqr':
                # Rolling Interquartile Range
                rolling_q1 = df[col].rolling(window=window, min_periods=10).quantile(0.25)
                rolling_q3 = df[col].rolling(window=window, min_periods=10).quantile(0.75)
                rolling_iqr = rolling_q3 - rolling_q1
                
                lower_bound = rolling_q1 - threshold * rolling_iqr
                upper_bound = rolling_q3 + threshold * rolling_iqr
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'ewma':
                # EWMA-based outlier detection (recent data weighted more)
                ewma = df[col].ewm(span=window, min_periods=10).mean()
                ewm_std = df[col].ewm(span=window, min_periods=10).std()
                
                # Avoid division by zero
                ewm_std = ewm_std.replace(0, np.nan)
                
                if ewm_std.isna().all():
                    continue
                
                outliers = np.abs(df[col] - ewma) > threshold * ewm_std
                
            elif method == 'mad':
                # Global Median Absolute Deviation (legacy)
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                
                if mad == 0:
                    continue
                
                modified_z_scores = 0.6745 * (df[col] - median) / mad
                outliers = np.abs(modified_z_scores) > threshold
                
            elif method == 'iqr':
                # Global Interquartile Range (legacy)
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            else:
                raise ValueError(f"Unknown method: {method}. Use 'rolling_mad', 'rolling_iqr', 'ewma', 'mad', or 'iqr'")
            
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
        
        method_desc = f"{method} (window={window})" if 'rolling' in method or method == 'ewma' else method
        logger.info(f"Outlier detection complete [{method_desc}]. Found {len(summary_df)} features with outliers")
        
        return outlier_mask, summary_df
    
    def winsorize_outliers(
        self, 
        df: pd.DataFrame, 
        limits: Tuple[float, float] = (0.01, 0.01),
        method: str = 'rolling',
        window: int = 60
    ) -> pd.DataFrame:
        """
        Winsorize outliers using TIME-SERIES AWARE methods.
        
        Improved methods for time series:
        - 'rolling': Rolling window percentiles (local baseline)
        - 'global': Global percentiles (legacy method)
        
        Args:
            df: DataFrame to process
            limits: (lower, upper) percentile limits
            method: 'rolling' or 'global'
            window: Rolling window size (for rolling method)
            
        Returns:
            DataFrame with winsorized values
        """
        df = df.copy()
        
        # Get numeric columns (exclude date_id and target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['date_id', 'forward_returns', 'risk_free_rate',
                       'market_forward_excess_returns', 'is_scored']
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        if method == 'rolling':
            # Rolling window winsorization (time-series aware)
            for col in numeric_cols:
                if df[col].notna().sum() > 0:
                    # Calculate rolling percentiles
                    rolling_lower = df[col].rolling(
                        window=window, 
                        min_periods=20, 
                        center=False
                    ).quantile(limits[0])
                    
                    rolling_upper = df[col].rolling(
                        window=window, 
                        min_periods=20, 
                        center=False
                    ).quantile(1 - limits[1])
                    
                    # Clip values based on rolling bounds
                    df[col] = df[col].clip(lower=rolling_lower, upper=rolling_upper)
            
            logger.info(f"Rolling winsorization complete with limits {limits}, window={window}")
            
        elif method == 'global':
            # Global winsorization (legacy method)
            for col in numeric_cols:
                if df[col].notna().sum() > 0:
                    df[col] = stats.mstats.winsorize(df[col].values, limits=limits)
            
            logger.info(f"Global winsorization complete with limits {limits}")
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'rolling' or 'global'")
        
        return df
    
    def detect_feature_clusters(
        self,
        df: pd.DataFrame,
        method: str = 'correlation',
        threshold: float = 0.85,
        by_group: bool = True
    ) -> Dict[str, List[List[str]]]:
        """
        Detect highly correlated feature clusters to identify redundancy.
        
        This helps identify groups of features that provide similar information,
        which can lead to overfitting in tree-based models or instability in linear models.
        
        Args:
            df: DataFrame to analyze
            method: 'correlation' (Pearson correlation) or 'distance' (not implemented)
            threshold: Correlation threshold for clustering (default: 0.85)
            by_group: Whether to cluster within feature groups (default: True)
            
        Returns:
            Dictionary mapping group name -> list of clusters
            Each cluster is a list of feature names
            
        Example:
            >>> clusters = loader.detect_feature_clusters(train_df, threshold=0.90)
            >>> # {'V': [['V1', 'V3', 'V7'], ['V2', 'V5']], 'M': [['M1', 'M4']]}
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        all_clusters = {}
        
        if by_group:
            # Cluster within each feature group
            groups_to_process = {k: v for k, v in self.feature_groups.items() if k != 'D'}
        else:
            # Cluster all features together
            all_features = [f for group_features in self.feature_groups.values() 
                          for f in group_features if f in df.columns]
            groups_to_process = {'all': all_features}
        
        for group, features in groups_to_process.items():
            group_features = [f for f in features if f in df.columns]
            
            if len(group_features) < 2:
                continue
            
            # Calculate correlation matrix
            corr_matrix = df[group_features].corr().abs()
            
            # Convert correlation to distance (1 - correlation)
            distance_matrix = 1 - corr_matrix
            
            # Perform hierarchical clustering
            try:
                # Convert to condensed distance matrix for linkage
                condensed_dist = squareform(distance_matrix, checks=False)
                
                # Perform clustering
                linkage_matrix = linkage(condensed_dist, method='average')
                
                # Cut tree at distance threshold
                distance_threshold = 1 - threshold
                cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
                
                # Group features by cluster
                clusters = {}
                for feature, label in zip(group_features, cluster_labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(feature)
                
                # Only keep clusters with multiple features
                group_clusters = [cluster for cluster in clusters.values() if len(cluster) > 1]
                
                if group_clusters:
                    all_clusters[group] = group_clusters
                    logger.info(f"Group {group}: Found {len(group_clusters)} clusters with {sum(len(c) for c in group_clusters)} features")
                    
            except Exception as e:
                logger.warning(f"Failed to cluster group {group}: {e}")
                continue
        
        return all_clusters
    
    def reduce_feature_redundancy(
        self,
        df: pd.DataFrame,
        clusters: Dict[str, List[List[str]]],
        method: str = 'representative'
    ) -> pd.DataFrame:
        """
        Reduce feature redundancy using clustering results.
        
        Methods:
        - 'representative': Keep only the feature with highest variance in each cluster
        - 'mean': Replace cluster with mean of z-scored features
        - 'pca': Replace cluster with first principal component (not implemented yet)
        
        Args:
            df: DataFrame to process
            clusters: Output from detect_feature_clusters()
            method: Reduction method
            
        Returns:
            DataFrame with reduced features
            
        Example:
            >>> clusters = loader.detect_feature_clusters(train_df, threshold=0.90)
            >>> df_reduced = loader.reduce_feature_redundancy(df, clusters, method='representative')
        """
        df = df.copy()
        features_removed = 0
        
        for group, group_clusters in clusters.items():
            for cluster in group_clusters:
                if len(cluster) < 2:
                    continue
                
                if method == 'representative':
                    # Keep feature with highest variance (most informative)
                    variances = df[cluster].var()
                    representative = variances.idxmax()
                    
                    # Drop all others in the cluster
                    to_drop = [f for f in cluster if f != representative]
                    df = df.drop(columns=to_drop)
                    features_removed += len(to_drop)
                    
                    logger.debug(f"Cluster {cluster}: Kept {representative}, dropped {len(to_drop)} features")
                    
                elif method == 'mean':
                    # Create mean of z-scored features
                    cluster_name = f"{group}_cluster_{cluster[0]}"
                    
                    # Z-score each feature in cluster
                    z_scored = df[cluster].apply(lambda x: (x - x.mean()) / x.std())
                    
                    # Take mean
                    df[cluster_name] = z_scored.mean(axis=1)
                    
                    # Drop original features
                    df = df.drop(columns=cluster)
                    features_removed += len(cluster)
                    
                    logger.debug(f"Cluster {cluster}: Created {cluster_name}, dropped {len(cluster)} features")
                    
                else:
                    raise ValueError(f"Unknown method: {method}. Use 'representative' or 'mean'")
        
        logger.info(f"Feature redundancy reduction: Removed {features_removed} features using '{method}' method")
        
        return df
    
    def apply_group_pca(
        self,
        df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        n_components: Optional[Dict[str, int]] = None,
        variance_threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply PCA within each feature group for dimensionality reduction.
        
        This reduces the number of features while preserving most of the variance,
        helping to prevent overfitting and improve model generalization.
        
        Args:
            df: DataFrame to transform
            train_df: Training data for fitting PCA (optional)
            n_components: Number of components per group (optional)
                Example: {'M': 3, 'V': 2} means 3 components for M group, 2 for V
            variance_threshold: Cumulative variance to retain (default: 0.95)
            
        Returns:
            Tuple of (transformed DataFrame, PCA models dictionary)
            
        Example:
            >>> n_components = {'M': 5, 'V': 3, 'E': 4}
            >>> df_pca, pca_models = loader.apply_group_pca(train_df, n_components=n_components)
            >>> # Apply same transformation to test data
            >>> test_pca, _ = loader.apply_group_pca(test_df, train_df=train_df, n_components=n_components)
        """
        from sklearn.decomposition import PCA
        
        df = df.copy()
        pca_models = {}
        
        # Get non-dummy feature groups
        groups_to_process = {k: v for k, v in self.feature_groups.items() if k != 'D'}
        
        for group, features in groups_to_process.items():
            group_features = [f for f in features if f in df.columns]
            
            if len(group_features) < 2:
                logger.info(f"Group {group}: Skipping PCA (only {len(group_features)} features)")
                continue
            
            # Determine number of components
            if n_components and group in n_components:
                n = n_components[group]
            else:
                # Auto-determine based on variance threshold
                n = None  # Will be determined by PCA
            
            # Fit PCA on training data or current data
            fit_df = train_df if train_df is not None else df
            fit_data = fit_df[group_features].dropna()
            
            if len(fit_data) == 0:
                logger.warning(f"Group {group}: No valid data for PCA fitting")
                continue
            
            # Initialize and fit PCA
            if n is None:
                # Determine n_components to reach variance threshold
                pca = PCA(n_components=variance_threshold, svd_solver='full')
            else:
                pca = PCA(n_components=min(n, len(group_features)))
            
            try:
                pca.fit(fit_data)
                
                # Transform the data
                transformed = pca.transform(df[group_features].fillna(0))
                
                # Create new column names
                pca_cols = [f"{group}_PC{i+1}" for i in range(pca.n_components_)]
                
                # Add to dataframe
                for i, col in enumerate(pca_cols):
                    df[col] = transformed[:, i]
                
                # Drop original features
                df = df.drop(columns=group_features)
                
                # Store model
                pca_models[group] = {
                    'model': pca,
                    'original_features': group_features,
                    'pca_features': pca_cols,
                    'variance_explained': pca.explained_variance_ratio_.sum()
                }
                
                logger.info(f"Group {group}: Reduced {len(group_features)} features to {pca.n_components_} "
                          f"components (variance: {pca.explained_variance_ratio_.sum():.2%})")
                
            except Exception as e:
                logger.error(f"Group {group}: PCA failed - {e}")
                continue
        
        return df, pca_models
    
    def calculate_regime_weights(
        self,
        df: pd.DataFrame,
        regime_col: str = 'regime',
        weight_map: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Calculate sample weights based on volatility regime.
        
        High volatility periods often contain more noise than signal,
        so we down-weight these samples during training.
        
        Default weights:
        - Low Vol: 1.0 (standard weight)
        - Normal: 1.0
        - High Vol: 0.5 (reduced weight due to noise)
        
        Args:
            df: DataFrame with regime column
            regime_col: Name of regime column (default: 'regime')
            weight_map: Custom weight mapping (optional)
                Example: {'low_vol': 1.0, 'normal': 1.0, 'high_vol': 0.3}
            
        Returns:
            Series of sample weights (same length as df)
            
        Example:
            >>> df = loader.detect_regime_changes(df)
            >>> sample_weights = loader.calculate_regime_weights(df)
            >>> # Use in model training: model.fit(X, y, sample_weight=sample_weights)
        """
        if regime_col not in df.columns:
            logger.warning(f"Regime column '{regime_col}' not found. Returning uniform weights.")
            return pd.Series(1.0, index=df.index)
        
        if weight_map is None:
            weight_map = {
                'low_vol': 1.0,
                'normal': 1.0,
                'high_vol': 0.5  # Down-weight high volatility periods
            }
        
        weights = df[regime_col].map(weight_map)
        
        # Fill unknown regimes with 1.0 (normal weight)
        weights = weights.fillna(1.0)
        
        logger.info(f"Regime-based sample weights calculated: {weights.value_counts().to_dict()}")
        
        return weights
    
    def preprocess_timeseries(
        self,
        df: pd.DataFrame,
        train_df: Optional[pd.DataFrame] = None,
        handle_missing: bool = True,
        handle_outliers: bool = True,
        scale: bool = True,
        missing_strategy: Optional[Dict[str, str]] = None,
        outlier_method: str = 'rolling_mad',
        outlier_threshold: float = 4.0,
        winsorize_method: str = 'rolling',
        winsorize_limits: Tuple[float, float] = (0.01, 0.01),
        scale_method: str = 'robust',
        window: int = 60
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete TIME-SERIES AWARE preprocessing pipeline.
        
        This pipeline ensures:
        1. Temporal ordering is respected (no future leakage)
        2. Local patterns are considered (rolling windows)
        3. Recent data is weighted more (EWMA)
        4. Regime changes are handled (adaptive methods)
        
        Args:
            df: DataFrame to process (must be sorted by date_id)
            train_df: Training data for fitting (optional)
            handle_missing: Whether to handle missing values
            handle_outliers: Whether to handle outliers
            scale: Whether to scale features
            missing_strategy: Custom missing value strategy
            outlier_method: Outlier detection method
            outlier_threshold: Outlier threshold
            winsorize_method: Winsorization method
            winsorize_limits: Winsorization limits
            scale_method: Scaling method
            window: Rolling window size
            
        Returns:
            Tuple of (processed DataFrame, metadata dict)
        """
        df = df.copy()
        metadata = {}
        
        logger.info("="*60)
        logger.info("Starting TIME-SERIES PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        # Step 1: Handle missing values
        if handle_missing:
            with Timer("Missing value handling", logger):
                df = self.handle_missing_values(df, train_df, missing_strategy)
                metadata['missing_handled'] = True
        
        # Step 2: Detect and handle outliers
        if handle_outliers:
            with Timer("Outlier detection and treatment", logger):
                # Detect outliers
                outlier_mask, outlier_summary = self.detect_outliers(
                    df, 
                    method=outlier_method,
                    threshold=outlier_threshold,
                    window=window
                )
                metadata['outlier_summary'] = outlier_summary
                
                # Winsorize outliers
                df = self.winsorize_outliers(
                    df,
                    limits=winsorize_limits,
                    method=winsorize_method,
                    window=window
                )
                metadata['outliers_handled'] = True
        
        # Step 3: Scale features
        scalers = None
        if scale:
            with Timer("Feature scaling", logger):
                df, scalers = self.scale_features(
                    df,
                    train_df=train_df,
                    method=scale_method,
                    by_group=True
                )
                metadata['scalers'] = scalers
                metadata['scaled'] = True
        
        logger.info("="*60)
        logger.info("TIME-SERIES PREPROCESSING COMPLETE")
        logger.info("="*60)
        
        return df, metadata
    
    def detect_regime_changes(
        self,
        df: pd.DataFrame,
        vol_window: int = 60,
        vol_threshold: Tuple[float, float] = (0.5, 1.5)
    ) -> pd.Series:
        """
        Detect market regime changes (high volatility / low volatility periods).
        
        Useful for:
        - Adaptive preprocessing (different strategies per regime)
        - Risk management
        - Model ensemble weighting
        
        Args:
            df: DataFrame with 'forward_returns' column
            vol_window: Window for volatility calculation
            vol_threshold: (low_multiplier, high_multiplier) for regime classification
            
        Returns:
            Series with regime labels: 'low_vol', 'normal', 'high_vol'
        """
        # Calculate rolling volatility
        returns_vol = df['forward_returns'].rolling(window=vol_window, min_periods=20).std()
        
        # Calculate median volatility as baseline
        vol_median = returns_vol.median()
        
        # Classify regimes
        regime = pd.Series('normal', index=df.index)
        regime[returns_vol > vol_threshold[1] * vol_median] = 'high_vol'
        regime[returns_vol < vol_threshold[0] * vol_median] = 'low_vol'
        
        # Count regime changes
        regime_changes = (regime != regime.shift(1)).sum()
        
        regime_counts = regime.value_counts()
        logger.info(f"Regime detection complete:")
        logger.info(f"  - Low volatility: {regime_counts.get('low_vol', 0)} periods")
        logger.info(f"  - Normal: {regime_counts.get('normal', 0)} periods")
        logger.info(f"  - High volatility: {regime_counts.get('high_vol', 0)} periods")
        logger.info(f"  - Regime changes: {regime_changes}")
        
        return regime
    
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
    # Test TIME-SERIES AWARE data loading and processing
    from utils import set_seed
    
    set_seed(42)
    
    # Initialize loader
    loader = DataLoader()
    
    # Load data
    train_df, test_df = loader.load_data()
    
    print("\n" + "="*60)
    print("BASIC DATA QUALITY CHECKS")
    print("="*60)
    
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
    
    print("\n" + "="*60)
    print("TIME-SERIES OUTLIER DETECTION COMPARISON")
    print("="*60)
    
    # Compare different outlier detection methods
    print("\n--- Global MAD (Legacy) ---")
    outlier_mask_global, outlier_summary_global = loader.detect_outliers(
        train_df, method='mad', threshold=4.0
    )
    print(f"Features with outliers: {len(outlier_summary_global)}")
    if len(outlier_summary_global) > 0:
        print(outlier_summary_global.head(5))
    
    print("\n--- Rolling MAD (Time-Series Aware) ---")
    outlier_mask_rolling, outlier_summary_rolling = loader.detect_outliers(
        train_df, method='rolling_mad', threshold=4.0, window=60
    )
    print(f"Features with outliers: {len(outlier_summary_rolling)}")
    if len(outlier_summary_rolling) > 0:
        print(outlier_summary_rolling.head(5))
    
    print("\n--- EWMA-based (Adaptive) ---")
    outlier_mask_ewma, outlier_summary_ewma = loader.detect_outliers(
        train_df, method='ewma', threshold=3.0, window=60
    )
    print(f"Features with outliers: {len(outlier_summary_ewma)}")
    if len(outlier_summary_ewma) > 0:
        print(outlier_summary_ewma.head(5))
    
    print("\n" + "="*60)
    print("REGIME CHANGE DETECTION")
    print("="*60)
    
    # Detect market regimes
    regime = loader.detect_regime_changes(train_df, vol_window=60)
    print(f"\nRegime distribution:\n{regime.value_counts()}")
    
    print("\n" + "="*60)
    print("TIME-SERIES PREPROCESSING PIPELINE TEST")
    print("="*60)
    
    # Test the complete preprocessing pipeline
    train_processed, metadata = loader.preprocess_timeseries(
        train_df,
        train_df=None,  # No separate train data (will use self for fitting)
        handle_missing=True,
        handle_outliers=True,
        scale=True,
        outlier_method='rolling_mad',
        outlier_threshold=4.0,
        winsorize_method='rolling',
        winsorize_limits=(0.01, 0.01),
        scale_method='robust',
        window=60
    )
    
    print("\n=== Preprocessing Metadata ===")
    print(f"Missing handled: {metadata.get('missing_handled', False)}")
    print(f"Outliers handled: {metadata.get('outliers_handled', False)}")
    print(f"Scaled: {metadata.get('scaled', False)}")
    if 'outlier_summary' in metadata:
        print(f"Features with outliers detected: {len(metadata['outlier_summary'])}")
    
    print("\n" + "="*60)
    print("BENCHMARK CALCULATION")
    print("="*60)
    
    # Calculate benchmark
    score, returns, metrics = calculate_benchmark_score(train_df, allocation=1.0)
    print("\n=== Benchmark Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "="*60)
    print("DATA COMPARISON: BEFORE vs AFTER PREPROCESSING")
    print("="*60)
    
    # Compare statistics before and after
    feature = 'M1'  # Example feature
    if feature in train_df.columns:
        print(f"\n=== Feature: {feature} ===")
        print(f"Before - Missing: {train_df[feature].isna().sum()}, "
              f"Mean: {train_df[feature].mean():.4f}, "
              f"Std: {train_df[feature].std():.4f}")
        print(f"After  - Missing: {train_processed[feature].isna().sum()}, "
              f"Mean: {train_processed[feature].mean():.4f}, "
              f"Std: {train_processed[feature].std():.4f}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
