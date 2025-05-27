"""Functions for detecting outliers in sales data."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

from src.config import QNT_THRESHOLD, ZSCORE_THRESHOLD

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_zscore_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate z-scores for 'qnt' within groups of index_store + index_material.
    
    Args:
        df: DataFrame containing sales data
        
    Returns:
        DataFrame with added 'zscore' column
    """
    logger.info("Calculating z-scores by store-product groups...")
    
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Define the group key
    result_df['group_key'] = result_df['index_store'].astype(str) + '_' + result_df['index_material'].astype(str)
    
    # Calculate z-scores within each group
    group_stats = result_df.groupby('group_key')['qnt'].agg(['mean', 'std']).reset_index()
    group_stats.rename(columns={'mean': 'group_mean', 'std': 'group_std'}, inplace=True)
    
    # Merge stats back to the original dataframe
    result_df = pd.merge(result_df, group_stats, on='group_key', how='left')
    
    # Calculate z-scores, handling cases where std=0
    mask = result_df['group_std'] > 0
    result_df['zscore'] = np.nan
    result_df.loc[mask, 'zscore'] = (result_df.loc[mask, 'qnt'] - result_df.loc[mask, 'group_mean']) / result_df.loc[mask, 'group_std']
    
    # Fill NaN z-scores with 0 (occurs when std=0)
    result_df['zscore'].fillna(0, inplace=True)
    
    logger.info("Z-score calculation complete")
    return result_df


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers based on z-score and quantity thresholds.
    
    Outliers are defined as:
    - qnt > QNT_THRESHOLD (default: 10)
    - |zscore| >= ZSCORE_THRESHOLD (default: 10)
    
    Args:
        df: DataFrame containing sales data
        
    Returns:
        DataFrame containing only the outlier rows
    """
    if df.empty:
        logger.warning("Empty dataframe provided for outlier detection")
        return pd.DataFrame()
    
    logger.info(f"Starting outlier detection with thresholds: qnt > {QNT_THRESHOLD}, |zscore| >= {ZSCORE_THRESHOLD}")
    
    # Calculate z-scores
    df_with_zscores = calculate_zscore_by_group(df)
    
    # Apply outlier criteria
    outlier_mask = (df_with_zscores['qnt'] > QNT_THRESHOLD) & (df_with_zscores['zscore'].abs() >= ZSCORE_THRESHOLD)
    outliers = df_with_zscores[outlier_mask].copy()
    
    # Add outlier flag
    outliers['is_outlier'] = True
    
    logger.info(f"Found {len(outliers)} outliers out of {len(df)} records ({len(outliers)/len(df)*100:.3f}%)")
    
    return outliers
