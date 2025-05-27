"""Functions for preprocessing sales data."""

import pandas as pd
import numpy as np
import os
import glob
from typing import List, Dict, Optional, Union, Tuple
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_monthly_data(data_dir: str, months: List[int] = None, year: int = 2023, 
                     clean_files: bool = True) -> pd.DataFrame:
    """Load monthly data files.
    
    Args:
        data_dir: Directory containing the data files
        months: List of months to load (1-12). If None, load all available months
        year: Year of data files
        clean_files: Whether to load files with '_clean' suffix from outlier removal
        
    Returns:
        DataFrame with combined monthly data
    """
    # First try to load clean files if requested
    files_to_load = []
    
    if clean_files:
        # Try to find cleaned files with proper pattern
        clean_pattern = f"{year}??_clean.csv"
        all_clean_files = sorted(glob.glob(os.path.join(data_dir, clean_pattern)))
        
        # Filter by months if specified
        if months and all_clean_files:
            filtered_clean_files = []
            for month in months:
                month_pattern = f"{year}{month:02d}_clean"
                for file in all_clean_files:
                    if month_pattern in os.path.basename(file):
                        filtered_clean_files.append(file)
            all_clean_files = sorted(filtered_clean_files)
        
        if all_clean_files:
            logger.info(f"Found {len(all_clean_files)} cleaned data files")
            files_to_load = all_clean_files
        else:
            logger.warning(f"No cleaned data files found in {data_dir} matching pattern {clean_pattern}")
            logger.info("Falling back to original files")
    
    # If no cleaned files found (or not requested), use original files
    if not files_to_load:
        original_pattern = f"{year}??.csv"
        all_files = sorted(glob.glob(os.path.join(data_dir, original_pattern)))
        
        # Filter out any clean files to avoid duplicates
        original_files = [f for f in all_files if not '_clean.csv' in f]
        
        # Filter by months if specified
        if months:
            filtered_files = []
            for month in months:
                month_pattern = f"{year}{month:02d}.csv"
                for file in original_files:
                    if month_pattern in os.path.basename(file):
                        filtered_files.append(file)
            original_files = sorted(filtered_files)
        
        files_to_load = original_files
    
    if not files_to_load:
        logger.warning(f"No data files found in {data_dir} matching requirements")
        return pd.DataFrame()
    
    logger.info(f"Found {len(files_to_load)} data files to load")
    
    # Initialize list to store DataFrames
    dfs = []
    
    # Load each file
    for file_path in files_to_load:
        logger.info(f"Loading {os.path.basename(file_path)}...")
        
        try:
            # First try with semicolon separator
            df = pd.read_csv(file_path, sep=';')
            
            # Extract month from filename for reference
            base_name = os.path.basename(file_path)
            month = base_name.replace('.csv', '').replace('_clean', '')[-2:]
            df['source_month'] = month
            
            # Handle calday format (convert to datetime) - direct approach with explicit format
            df['calday'] = pd.to_datetime(df['calday'].astype(str), format='%Y%m%d', errors='coerce')
            
            dfs.append(df)
            logger.info(f"Loaded {len(df)} rows from {os.path.basename(file_path)}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    
    # Combine all data frames
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined data has {len(combined_df)} rows and {len(combined_df.columns)} columns")
        return combined_df
    else:
        logger.warning("No data was successfully loaded")
        return pd.DataFrame()

def create_hierarchical_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Create hierarchical category features from subcat_id.
    
    Args:
        df: DataFrame containing subcat_id column
        
    Returns:
        DataFrame with added hierarchical category features
    """
    if 'subcat_id' not in df.columns:
        logger.warning("subcat_id column not found in DataFrame")
        return df
    
    # Make a copy to avoid modifying the original
    df_cats = df.copy()
    
    # Convert subcat_id to string to ensure proper slicing
    df_cats['subcat_id'] = df_cats['subcat_id'].astype(str)
    
    # Create hierarchical categories
    # - First 2 digits: major category (most general)
    # - First 4 digits: detailed category (medium detail)
    # - All 6 digits: most detailed category (already in subcat_id)
    
    def extract_category(row, digits):
        # Pad with zeros if needed
        cat_str = row.zfill(6)[:digits]
        return cat_str
    
    df_cats['category_major'] = df_cats['subcat_id'].apply(lambda x: extract_category(x, 2))
    df_cats['category_detailed'] = df_cats['subcat_id'].apply(lambda x: extract_category(x, 4))
    df_cats['category_full'] = df_cats['subcat_id']
    
    # Convert back to integer for efficiency
    df_cats['category_major'] = pd.to_numeric(df_cats['category_major'], errors='coerce').fillna(0).astype(int)
    df_cats['category_detailed'] = pd.to_numeric(df_cats['category_detailed'], errors='coerce').fillna(0).astype(int)
    
    logger.info("Created hierarchical category features")
    return df_cats

def create_date_features(df: pd.DataFrame, date_col: str = 'calday') -> pd.DataFrame:
    """Create date-related features.
    
    Args:
        df: DataFrame with date column
        date_col: Name of the date column
        
    Returns:
        DataFrame with added date features
    """
    if date_col not in df.columns:
        logger.warning(f"{date_col} column not found in DataFrame")
        return df
    
    # Make a copy to avoid modifying the original
    df_dates = df.copy()
    
    # Ensure date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_dates[date_col]):
        # Check the data type and format
        sample = df_dates[date_col].iloc[0] if not df_dates[date_col].isna().all() else None
        
        logger.info(f"Converting {date_col} to datetime. Current type: {type(sample)}, Sample value: {sample}")
        
        df_dates[date_col] = df_dates[date_col].astype(str).str

        # Use explicit YYYYMMDD format for parsing
        df_dates[date_col] = pd.to_datetime(
            df_dates[date_col], 
            format='%Y%m%d',
            errors='coerce'
        )

    df_dates['day_of_week'] = df_dates[date_col].dt.dayofweek
    df_dates['day_of_month'] = df_dates[date_col].dt.day
    df_dates['month'] = df_dates[date_col].dt.month
    df_dates['quarter'] = df_dates[date_col].dt.quarter
    df_dates['year'] = df_dates[date_col].dt.year
    df_dates['is_weekend'] = df_dates['day_of_week'].isin([5, 6])  # 5=Saturday, 6=Sunday
    df_dates['is_month_start'] = df_dates[date_col].dt.is_month_start
    df_dates['is_month_end'] = df_dates[date_col].dt.is_month_end
    
    # Add season feature
    season_map = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    }
    df_dates['season'] = df_dates['month'].map(season_map)
    
    # Add cyclical encoding for day of week, day of month, month
    df_dates['day_of_week_sin'] = np.sin(2 * np.pi * df_dates['day_of_week'] / 7)
    df_dates['day_of_week_cos'] = np.cos(2 * np.pi * df_dates['day_of_week'] / 7)
    df_dates['day_of_month_sin'] = np.sin(2 * np.pi * df_dates['day_of_month'] / 31)
    df_dates['day_of_month_cos'] = np.cos(2 * np.pi * df_dates['day_of_month'] / 31)
    df_dates['month_sin'] = np.sin(2 * np.pi * df_dates['month'] / 12)
    df_dates['month_cos'] = np.cos(2 * np.pi * df_dates['month'] / 12)
    
    logger.info("Created date features")
    return df_dates

def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create price-related features.
    
    Args:
        df: DataFrame with price columns
        
    Returns:
        DataFrame with added price features
    """
    # Make a copy to avoid modifying the original
    df_prices = df.copy()
    
    # Create price ratio (action to regular)
    if 'action_price' in df_prices.columns and 'regular_price' in df_prices.columns:
        # Calculate price ratio
        df_prices['price_ratio'] = df_prices['action_price'] / df_prices['regular_price']
        # Handle potential division by zero or invalid values
        df_prices['price_ratio'] = df_prices['price_ratio'].replace([np.inf, -np.inf], np.nan)
        # Ensure ratio is between 0 and 1 (discount shouldn't increase price)
        df_prices['price_ratio'] = np.clip(df_prices['price_ratio'].fillna(1), 0, 1)
        
        # Calculate price difference (regular - action)
        df_prices['price_diff'] = df_prices['regular_price'] - df_prices['action_price']
        df_prices['price_diff'] = df_prices['price_diff'].fillna(0)
        df_prices['price_diff'] = np.maximum(df_prices['price_diff'], 0)  # Shouldn't be negative
        
        # Calculate price discount percentage
        df_prices['price_discount_pct'] = 1 - df_prices['price_ratio']
        df_prices['price_discount_pct'] = np.clip(df_prices['price_discount_pct'], 0, 1)
        
        logger.info("Created price comparison features")
    else:
        logger.warning("Missing action_price or regular_price columns for price features")
    
    return df_prices

def create_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create promotion-related features.
    
    Args:
        df: DataFrame with promotion columns
        
    Returns:
        DataFrame with added promotion features
    """
    # Make a copy to avoid modifying the original
    df_promo = df.copy()
    
    # Identify promotions based on discount, price difference, or promo dates
    conditions = []
    if 'discount' in df_promo.columns:
        mask_discount = (~df_promo['discount'].isna()) & (df_promo['discount'] > 0)
        conditions.append(mask_discount)
    
    if all(col in df_promo.columns for col in ['regular_price', 'action_price']):
        mask_price_diff = (~df_promo['regular_price'].isna()) & (~df_promo['action_price'].isna()) & (df_promo['regular_price'] > df_promo['action_price'])
        conditions.append(mask_price_diff)
    
    if all(col in df_promo.columns for col in ['promo_from', 'promo_to']):
        mask_promo_dates = (~df_promo['promo_from'].isna()) & (~df_promo['promo_to'].isna())
        conditions.append(mask_promo_dates)
    
    # Create is_promo flag if we have any conditions
    if conditions:
        df_promo['is_promo'] = False
        for condition in conditions:
            df_promo['is_promo'] = df_promo['is_promo'] | condition
        
        logger.info(f"Identified {df_promo['is_promo'].sum()} promotion records ({df_promo['is_promo'].mean()*100:.2f}%)")
    else:
        logger.warning("Missing necessary columns for promotion detection")
        df_promo['is_promo'] = False
    
    # Handle promo dates and calculate promo-related features
    if all(col in df_promo.columns for col in ['calday', 'promo_from', 'promo_to']):
        # Ensure date columns are in datetime format
        date_cols = ['calday', 'promo_from', 'promo_to']
        for col in date_cols:
            if not pd.api.types.is_datetime64_any_dtype(df_promo[col]):
                df_promo[col] = pd.to_datetime(df_promo[col], errors='coerce')
        
        # For promotions without explicit dates, use calday
        mask = (df_promo['is_promo']) & (df_promo['promo_from'].isna())
        df_promo.loc[mask, 'promo_from'] = df_promo.loc[mask, 'calday']
        
        mask = (df_promo['is_promo']) & (df_promo['promo_to'].isna())
        df_promo.loc[mask, 'promo_to'] = df_promo.loc[mask, 'calday']
        
        # Calculate days since promo start and until promo end
        PROMO_MISSING_VALUE = -1  # Use -1 to indicate no promo
        
        df_promo['days_since_promo_start'] = (df_promo['calday'] - df_promo['promo_from']).dt.days
        df_promo['days_until_promo_end'] = (df_promo['promo_to'] - df_promo['calday']).dt.days
        
        # Fill missing values with the designated missing value
        df_promo['days_since_promo_start'] = df_promo['days_since_promo_start'].fillna(PROMO_MISSING_VALUE)
        df_promo['days_until_promo_end'] = df_promo['days_until_promo_end'].fillna(PROMO_MISSING_VALUE)
        
        # Create flags for promo status
        df_promo['is_before_promo'] = df_promo['days_since_promo_start'] < 0
        df_promo['is_after_promo'] = df_promo['days_until_promo_end'] < 0
        df_promo['is_during_promo'] = (df_promo['days_since_promo_start'] >= 0) & (df_promo['days_until_promo_end'] >= 0)
        df_promo['is_no_promo'] = df_promo['days_since_promo_start'] == PROMO_MISSING_VALUE
        
        # Calculate promo duration in days
        df_promo['promo_duration'] = (df_promo['promo_to'] - df_promo['promo_from']).dt.days
        df_promo['promo_duration'] = df_promo['promo_duration'].fillna(0).astype(int)
        
        logger.info("Created promotion timing features")
    
    return df_promo

def create_lag_features(
    df: pd.DataFrame,
    target_col: str = 'qnt',
    group_cols: List[str] = ['index_store', 'index_material'],
    lag_days: List[int] = [14, 21, 28, 35, 42],
    date_col: str = 'calday'
) -> pd.DataFrame:
    """Create lag features that respect the 14+ day forecast horizon.
    
    Args:
        df: DataFrame with sales data
        target_col: Column to create lags for (usually 'qnt')
        group_cols: Columns to group by (usually store and product)
        lag_days: List of days to lag (must be >= 14)
        date_col: Name of the date column
        
    Returns:
        DataFrame with added lag features
    """
    # Validate required columns exist
    required_cols = group_cols + [target_col, date_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns for lag features: {missing_cols}")
        return df
    
    # Validate that all lag days are at least 14 days
    if any(lag < 14 for lag in lag_days):
        logger.warning("Some lag days are less than 14 days. These will be removed.")
        lag_days = [lag for lag in lag_days if lag >= 14]
        if not lag_days:
            logger.warning("No valid lag days >= 14 provided")
            return df
    
    logger.info(f"Creating lag features (14+ days) for '{target_col}'")
    
    # Make a copy to avoid modifying the original
    df_lag = df.copy()
    
    # Ensure date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df_lag[date_col]):
        df_lag[date_col] = pd.to_datetime(df_lag[date_col], errors='coerce')
    
    # First, aggregate by store, product, and date in case of multiple records per day
    logger.info("Aggregating data by store, product, and date...")
    agg_df = df_lag.groupby(group_cols + [date_col])[target_col].sum().reset_index()
    
    # Sort by store, product, and date to ensure proper lag calculations
    agg_df = agg_df.sort_values(group_cols + [date_col])
    
    # Create lag features
    lag_cols = []
    for lag in lag_days:
        lag_col = f'{target_col}_lag_{lag}d'
        lag_cols.append(lag_col)
        
        # Calculate lags for each group
        agg_df[lag_col] = agg_df.groupby(group_cols)[target_col].shift(lag)
        logger.info(f"Created {lag_col}")
    
    # Calculate average of lags
    avg_col = f'{target_col}_lag_avg'
    agg_df[avg_col] = agg_df[lag_cols].mean(axis=1)
    lag_cols.append(avg_col)
    logger.info(f"Created {avg_col} as average of individual lags")
    
    # Keep only necessary columns for merging back to the main dataframe
    lag_features = agg_df[group_cols + [date_col] + lag_cols]
    
    # Merge lag features back to the main DataFrame
    logger.info("Merging lag features back to main dataframe...")
    df_lag = pd.merge(df_lag, lag_features, on=group_cols + [date_col], how='left')
    
    # Fill missing values in lag features
    for col in lag_cols:
        # Step 1: Forward and backward fill within groups
        df_lag[col] = df_lag.groupby(group_cols)[col].transform(lambda x: x.ffill().bfill())
        
        # Check if we still have nulls
        nulls_remaining = df_lag[col].isna().sum()
        if nulls_remaining > 0:
            # Step 2: Fill remaining nulls with product median across all stores
            product_col = group_cols[1]  # Assuming index_material is the second group column
            product_medians = df_lag.groupby(product_col)[col].transform('median')
            df_lag[col] = df_lag[col].fillna(product_medians)
            
            # Step 3: Fill with overall median if still nulls remain
            nulls_remaining = df_lag[col].isna().sum()
            if nulls_remaining > 0:
                df_lag[col] = df_lag[col].fillna(df_lag[col].median())
                
                # Step 4: Last resort - fill with 0
                df_lag[col] = df_lag[col].fillna(0)
    
    logger.info("Lag features creation complete")
    return df_lag

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the DataFrame.
    
    Args:
        df: DataFrame with possibly missing values
        
    Returns:
        DataFrame with filled missing values
    """
    logger.info("Handling missing values")
    
    # Make a copy to avoid modifying the original
    df_filled = df.copy()
    
    # Check for columns with missing values
    missing_counts = df_filled.isnull().sum()
    missing_percent = missing_counts / len(df_filled) * 100
    
    # Display columns with significant missing values (>1%)
    significant_missing = missing_percent[missing_percent > 1].sort_values(ascending=False)
    if len(significant_missing) > 0:
        logger.info("Columns with significant missing values (>1%):")
        for col, pct in significant_missing.items():
            logger.info(f"  {col}: {pct:.2f}% missing ({missing_counts[col]} values)")
    
    # 1. Handle missing qnt values
    if 'qnt' in df_filled.columns:
        missing_qnt = df_filled['qnt'].isna().sum()
        if missing_qnt > 0:
            logger.info(f"Filling {missing_qnt} missing qnt values with 0")
            df_filled['qnt'] = df_filled['qnt'].fillna(0)
    
    # 2. Handle missing type_bonus_id values
    if 'type_bonus_id' in df_filled.columns:
        # For non-promo items (if is_promo exists), use 'no_bonus'
        if 'is_promo' in df_filled.columns:
            mask = (~df_filled['is_promo']) & (df_filled['type_bonus_id'].isna())
            df_filled.loc[mask, 'type_bonus_id'] = 'no_bonus'
            
            # For promo items, check bu_exists
            if 'bu_exists' in df_filled.columns:
                # For promo items without a bonus (bu_exists = 0), use 'promo_no_bonus'
                mask = (df_filled['is_promo']) & (df_filled['type_bonus_id'].isna()) & (df_filled['bu_exists'] == 0)
                df_filled.loc[mask, 'type_bonus_id'] = 'promo_no_bonus'
                
                # For promo items with a bonus (bu_exists = 1), use 'unknown'
                mask = (df_filled['is_promo']) & (df_filled['type_bonus_id'].isna()) & (df_filled['bu_exists'] == 1)
                df_filled.loc[mask, 'type_bonus_id'] = 'unknown'
        
        # Any remaining NaNs, fill with 'unknown'
        df_filled['type_bonus_id'] = df_filled['type_bonus_id'].fillna('unknown')
        logger.info("Filled missing type_bonus_id values")
    
    # 3. Handle missing discount values
    if 'discount' in df_filled.columns:
        # If is_promo exists, set discount to 0 for non-promo items
        if 'is_promo' in df_filled.columns:
            mask = (~df_filled['is_promo']) & (df_filled['discount'].isna())
            df_filled.loc[mask, 'discount'] = 0
        
        # For items with regular and action prices, calculate discount
        if all(col in df_filled.columns for col in ['regular_price', 'action_price']):
            mask = (df_filled['discount'].isna()) & (~df_filled['regular_price'].isna()) & (~df_filled['action_price'].isna()) & (df_filled['regular_price'] > df_filled['action_price'])
            df_filled.loc[mask, 'discount'] = 1 - (df_filled.loc[mask, 'action_price'] / df_filled.loc[mask, 'regular_price'])
        
        # Fill remaining NaNs with 0 (assuming no discount)
        df_filled['discount'] = df_filled['discount'].fillna(0)
        logger.info("Filled missing discount values")
    
    # 4. Handle missing price columns
    price_columns = [
        'regular_price', 'action_price',
        'reg_mediana_price', 'action_mediana_price',
        'last_reg_mediana_price', 'last_action_mediana_price'
    ]
    
    # First, handle relationships between price columns
    if all(col in df_filled.columns for col in ['regular_price', 'action_price']):
        # If regular_price is missing but action_price is available and no promotion,
        # regular_price = action_price
        if 'is_promo' in df_filled.columns:
            mask = (~df_filled['is_promo']) & (df_filled['regular_price'].isna()) & (~df_filled['action_price'].isna())
            df_filled.loc[mask, 'regular_price'] = df_filled.loc[mask, 'action_price']
        
        # If action_price is missing but regular_price and discount are available,
        # calculate action_price
        if 'discount' in df_filled.columns:
            mask = (df_filled['action_price'].isna()) & (~df_filled['regular_price'].isna()) & (~df_filled['discount'].isna()) & (df_filled['discount'] > 0)
            df_filled.loc[mask, 'action_price'] = df_filled.loc[mask, 'regular_price'] * (1 - df_filled.loc[mask, 'discount'])
    
    # Fill price columns by using related columns and then by category
    for col in price_columns:
        if col in df_filled.columns:
            # Use related columns if available
            if col == 'regular_price' and 'last_reg_mediana_price' in df_filled.columns:
                mask = (df_filled[col].isna()) & (~df_filled['last_reg_mediana_price'].isna())
                df_filled.loc[mask, col] = df_filled.loc[mask, 'last_reg_mediana_price']
            
            elif col == 'regular_price' and 'reg_mediana_price' in df_filled.columns:
                mask = (df_filled[col].isna()) & (~df_filled['reg_mediana_price'].isna())
                df_filled.loc[mask, col] = df_filled.loc[mask, 'reg_mediana_price']
            
            elif col == 'action_price' and 'last_action_mediana_price' in df_filled.columns:
                mask = (df_filled[col].isna()) & (~df_filled['last_action_mediana_price'].isna())
                df_filled.loc[mask, col] = df_filled.loc[mask, 'last_action_mediana_price']
            
            elif col == 'action_price' and 'action_mediana_price' in df_filled.columns:
                mask = (df_filled[col].isna()) & (~df_filled['action_mediana_price'].isna())
                df_filled.loc[mask, col] = df_filled.loc[mask, 'action_mediana_price']
            
            # Fill by product
            if 'index_material' in df_filled.columns:
                # Group by product and fill with median
                product_medians = df_filled.groupby('index_material')[col].transform('median')
                df_filled[col] = df_filled[col].fillna(product_medians)
            
            # Fill by category if available
            if 'category_detailed' in df_filled.columns:
                category_medians = df_filled.groupby('category_detailed')[col].transform('median')
                df_filled[col] = df_filled[col].fillna(category_medians)
            
            # Fill with overall median as last resort
            overall_median = df_filled[col].median()
            if pd.notna(overall_median):
                df_filled[col] = df_filled[col].fillna(overall_median)
            else:
                # If even the overall median is NaN, use 0
                df_filled[col] = df_filled[col].fillna(0)
    
    logger.info("Filled missing price values")
    
    # 5. Handle remaining numeric columns with 0
    numeric_cols = df_filled.select_dtypes(include=['number']).columns.tolist()
    for col in numeric_cols:
        if df_filled[col].isna().sum() > 0:
            df_filled[col] = df_filled[col].fillna(0)
    
    # 6. Handle remaining categorical columns with 'unknown'
    categorical_cols = df_filled.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in categorical_cols:
        if df_filled[col].isna().sum() > 0:
            df_filled[col] = df_filled[col].fillna('unknown')
    
    # Check for any remaining missing values
    remaining_missing = df_filled.isnull().sum()
    remaining_cols = remaining_missing[remaining_missing > 0]
    if len(remaining_cols) > 0:
        logger.warning("There are still missing values in the following columns:")
        for col, count in remaining_cols.items():
            logger.warning(f"  {col}: {count} missing values")
    else:
        logger.info("All missing values handled successfully")
    
    return df_filled

def create_sales_performance_metrics(df: pd.DataFrame, 
                                     group_cols: List[str] = ['index_store', 'index_material']) -> pd.DataFrame:
    """Create sales performance metrics like qnt_max and qnt_percent.
    
    Args:
        df: DataFrame containing sales data
        group_cols: Columns to group by (usually store and product identifiers)
        
    Returns:
        DataFrame with added sales performance metrics
    """
    if 'qnt' not in df.columns:
        logger.warning("qnt column not found in DataFrame, can't create sales performance metrics")
        return df
    
    # Make a copy to avoid modifying the original
    df_metrics = df.copy()
    
    # Calculate max sales for each product in each store
    df_metrics['qnt_max'] = df_metrics.groupby(group_cols)['qnt'].transform('max')
    
    # Calculate sales percentage (relative to max)
    # When qnt_max is 0, set qnt_percent to 0 to avoid division by zero
    df_metrics['qnt_percent'] = np.where(df_metrics['qnt_max'] > 0, 
                                       df_metrics['qnt'] / df_metrics['qnt_max'], 
                                       0)
    
    # Handle any potential NaNs or infinities
    df_metrics['qnt_percent'] = df_metrics['qnt_percent'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    logger.info("Created sales performance metrics (qnt_max and qnt_percent)")
    return df_metrics

def process_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process sales data through the complete pipeline.
    
    Args:
        df: Raw sales DataFrame
        
    Returns:
        Processed DataFrame
    """
    logger.info("Starting sales data processing pipeline")
    
    # 1. Create hierarchical category features
    df = create_hierarchical_categories(df)
    
    # 2. Create date features
    df = create_date_features(df)
    
    # 3. Create promotion features
    df = create_promotion_features(df)
    
    # 4. Create price features
    df = create_price_features(df)
    
    # 5. Handle missing values
    df = handle_missing_values(df)
    
    # 6. Create lag features (14+ days)
    df = create_lag_features(df)
    
    # 7. Create sales performance metrics
    df = create_sales_performance_metrics(df)
    
    logger.info("Sales data processing pipeline complete")
    return df
