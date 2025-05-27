"""Functions for creating advanced weather features."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_lag_features(
    df: pd.DataFrame, 
    features: List[str],
    lag_days: List[int] = [14, 15, 16, 21, 28, 35, 42],  # Only lags of 14+ days
    date_col: str = 'calday'
) -> pd.DataFrame:
    """Create lag features for specified columns, respecting forecast horizon.
    
    Only creates lags for 14+ days to avoid using data that wouldn't be available
    at prediction time for the 2-15 day forecast horizon.
    
    Args:
        df: DataFrame with weather data
        features: List of columns to create lags for
        lag_days: List of days to lag (must be >= 14)
        date_col: Name of the date column if not using a datetime index
        
    Returns:
        DataFrame with added lag features
    """
    # Validate that all lag days are at least 14 days
    if any(lag < 14 for lag in lag_days):
        logger.warning("Some lag days are less than 14 days. These may not be available at prediction time.")
        # Filter to only include lags >= 14 days
        lag_days = [lag for lag in lag_days if lag >= 14]
        
    logger.info(f"Creating lag features (14+ days) for {len(features)} columns")
    
    # Make a copy to avoid modifying the original
    df_with_lags = df.copy()
    
    # If date_col exists in the DataFrame, set it as index temporarily
    has_date_col = date_col in df_with_lags.columns
    if has_date_col:
        df_with_lags = df_with_lags.set_index(date_col)
    
    # Check which features exist in the DataFrame
    available_features = [f for f in features if f in df_with_lags.columns]
    
    # Create lag features
    for feature in available_features:
        for lag in lag_days:
            lag_col = f"{feature}_lag_{lag}d"
            df_with_lags[lag_col] = df_with_lags[feature].shift(lag)
    
    # Reset index if we set it earlier
    if has_date_col:
        df_with_lags = df_with_lags.reset_index()
    
    logger.info(f"Created {len(available_features) * len(lag_days)} lag features")
    return df_with_lags

def create_forecast_safe_rolling_features(
    df: pd.DataFrame, 
    features: List[str],
    windows: List[Tuple[int, int]] = [(14, 28), (14, 21), (21, 35), (28, 42)],
    date_col: str = 'calday'
) -> pd.DataFrame:
    """Create rolling features that are safe to use for forecasting.
    
    Uses windows that look at data from at least 14 days ago to respect
    the forecast horizon of 2-15 days.
    
    Args:
        df: DataFrame with weather data
        features: List of columns to create rolling averages for
        windows: List of (start_day, end_day) tuples defining windows of past days
        date_col: Name of the date column if not using a datetime index
        
    Returns:
        DataFrame with added rolling features
    """
    logger.info(f"Creating forecast-safe rolling features for {len(features)} columns")
    
    # Make a copy to avoid modifying the original
    df_with_rolling = df.copy()
    
    # If date_col exists in the DataFrame, set it as index temporarily
    has_date_col = date_col in df_with_rolling.columns
    if has_date_col:
        df_with_rolling = df_with_rolling.set_index(date_col)
    
    # Check which features exist in the DataFrame
    available_features = [f for f in features if f in df_with_rolling.columns]
    
    # Create rolling window features
    for feature in available_features:
        for start_day, end_day in windows:
            # Calculate the window size
            window_size = end_day - start_day + 1
            
            # Create the feature name describing the window
            window_name = f"{feature}_d{start_day}to{end_day}"
            
            # Create a temporary shifted series for each day in the window
            temp_series = []
            for shift in range(start_day, end_day + 1):
                temp_series.append(df_with_rolling[feature].shift(shift))
            
            # Calculate statistics across the window
            df_with_rolling[f"{window_name}_mean"] = pd.concat(temp_series, axis=1).mean(axis=1)
            df_with_rolling[f"{window_name}_min"] = pd.concat(temp_series, axis=1).min(axis=1) 
            df_with_rolling[f"{window_name}_max"] = pd.concat(temp_series, axis=1).max(axis=1)
            df_with_rolling[f"{window_name}_std"] = pd.concat(temp_series, axis=1).std(axis=1)
    
    # Reset index if we set it earlier
    if has_date_col:
        df_with_rolling = df_with_rolling.reset_index()
    
    logger.info(f"Created forecast-safe rolling features")
    return df_with_rolling

def create_seasonal_features(df: pd.DataFrame, date_col: str = 'calday') -> pd.DataFrame:
    """Create seasonal features based on historical patterns for the same calendar day.
    
    These features calculate statistics from previous years for the same day of year,
    which is safe for forecasting as it doesn't use recent data.
    
    Args:
        df: DataFrame with weather data spanning multiple years
        date_col: Name of the date column
        
    Returns:
        DataFrame with added seasonal features
    """
    logger.info("Creating seasonal features")
    
    # Make a copy to avoid modifying the original
    df_seasonal = df.copy()
    
    # Ensure the date column is in datetime format
    if date_col in df_seasonal.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_seasonal[date_col]):
            df_seasonal[date_col] = pd.to_datetime(df_seasonal[date_col])
        
        # Extract day of year
        df_seasonal['day_of_year'] = df_seasonal[date_col].dt.dayofyear
        
        # If we have multiple years of data, we can calculate seasonal features
        if df_seasonal[date_col].dt.year.nunique() > 1:
            # Get key weather features
            weather_features = ['temp', 'tempmax', 'tempmin', 'precip', 'snow', 'cloudcover']
            weather_features = [f for f in weather_features if f in df_seasonal.columns]
            
            for feature in weather_features:
                # Group by day of year and calculate stats across years
                seasonal_stats = df_seasonal.groupby('day_of_year')[feature].agg(['mean', 'min', 'max', 'std']).reset_index()
                
                # Merge back to main dataframe
                df_seasonal = pd.merge(
                    df_seasonal,
                    seasonal_stats,
                    on='day_of_year',
                    suffixes=('', f'_seasonal')
                )
                
                # Rename columns for clarity
                df_seasonal.rename(columns={
                    'mean': f'{feature}_seasonal_mean',
                    'min': f'{feature}_seasonal_min',
                    'max': f'{feature}_seasonal_max',
                    'std': f'{feature}_seasonal_std'
                }, inplace=True)
                
    logger.info("Created seasonal features")
    return df_seasonal

def create_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temperature-related features.
    
    Args:
        df: DataFrame with weather data
        
    Returns:
        DataFrame with added temperature features
    """
    logger.info("Creating temperature features")
    
    # Make a copy to avoid modifying the original
    df_temp = df.copy()
    
    # Temperature delta (daily range)
    if 'tempmax' in df_temp.columns and 'tempmin' in df_temp.columns:
        df_temp['temp_range'] = df_temp['tempmax'] - df_temp['tempmin']
    
    # Freezing day flags
    if 'tempmin' in df_temp.columns:
        df_temp['freezing_day'] = df_temp['tempmin'] < 0
    
    if 'temp' in df_temp.columns:
        df_temp['cold_day'] = df_temp['temp'] < 0
        df_temp['warm_day'] = df_temp['temp'] > 15
        df_temp['hot_day'] = df_temp['temp'] > 25
    
    # Comfort metrics
    if 'temp' in df_temp.columns and 'humidity' in df_temp.columns:
        # Heat index (feels hotter when humid)
        mask = (df_temp['temp'] > 20)
        df_temp['heat_index'] = df_temp['temp'].copy()
        
        # Simple heat index formula
        df_temp.loc[mask, 'heat_index'] = df_temp.loc[mask, 'temp'] + 0.05 * df_temp.loc[mask, 'humidity']
    
    logger.info("Created temperature features")
    return df_temp

def create_past_extreme_indicators(
    df: pd.DataFrame, 
    features: List[str] = ['temp', 'precip', 'windspeed'],
    lookback_days: List[int] = [14, 28, 42], 
    date_col: str = 'calday'
) -> pd.DataFrame:
    """Create indicators for extreme weather events in the past periods.
    
    All lookback periods start from at least 14 days ago to respect the forecast horizon.
    
    Args:
        df: DataFrame with weather data
        features: Weather features to check for extremes
        lookback_days: End of lookback periods (all starting from 14 days ago)
        date_col: Name of the date column
        
    Returns:
        DataFrame with extreme weather indicators
    """
    logger.info("Creating past extreme weather indicators")
    
    # Make a copy to avoid modifying the original
    df_extremes = df.copy()
    
    # If date_col exists in the DataFrame, set it as index temporarily
    has_date_col = date_col in df_extremes.columns
    if has_date_col:
        df_extremes = df_extremes.set_index(date_col)
    
    # Define extreme thresholds for different features
    thresholds = {
        'temp': {'high': 25, 'low': 0},
        'tempmax': {'high': 30, 'low': 5},
        'tempmin': {'high': 20, 'low': -5},
        'precip': {'high': 10, 'low': None},
        'snow': {'high': 5, 'low': None},
        'windspeed': {'high': 30, 'low': None}
    }
    
    # Available features that exist in the dataframe and thresholds
    available_features = [f for f in features if f in df_extremes.columns and f in thresholds]
    
    # Create extreme indicators
    for feature in available_features:
        for lookback in lookback_days:
            # Check if feature exists in thresholds
            if feature in thresholds:
                # Create temporary series for days 14 to lookback
                temp_series = []
                for shift in range(14, lookback + 1):
                    temp_series.append(df_extremes[feature].shift(shift))
                
                if not temp_series:
                    continue
                
                # Combine into a DataFrame for easier analysis
                lookback_data = pd.concat(temp_series, axis=1)
                
                # Check if high threshold exists and create indicator
                if thresholds[feature]['high'] is not None:
                    high_threshold = thresholds[feature]['high']
                    # Did any day exceed the high threshold?
                    df_extremes[f"{feature}_had_high_d14to{lookback}"] = (
                        lookback_data > high_threshold
                    ).any(axis=1)
                    
                    # Count days exceeding threshold
                    df_extremes[f"{feature}_high_days_d14to{lookback}"] = (
                        lookback_data > high_threshold
                    ).sum(axis=1)
                
                # Check if low threshold exists and create indicator
                if thresholds[feature]['low'] is not None:
                    low_threshold = thresholds[feature]['low']
                    # Did any day go below the low threshold?
                    df_extremes[f"{feature}_had_low_d14to{lookback}"] = (
                        lookback_data < low_threshold
                    ).any(axis=1)
                    
                    # Count days below threshold
                    df_extremes[f"{feature}_low_days_d14to{lookback}"] = (
                        lookback_data < low_threshold
                    ).sum(axis=1)
    
    # Reset index if we set it earlier
    if has_date_col:
        df_extremes = df_extremes.reset_index()
    
    logger.info("Created past extreme weather indicators")
    return df_extremes

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in the DataFrame.
    
    Args:
        df: DataFrame with possibly missing values
        
    Returns:
        DataFrame with filled missing values
    """
    logger.info("Filling missing values")
    
    # Make a copy to avoid modifying the original
    df_filled = df.copy()
    
    # Check for columns with missing values
    missing_counts = df_filled.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0].index.tolist()
    
    # Fill missing values appropriately for each column type
    for col in cols_with_missing:
        # For lag and rolling features, get the base feature name
        if '_lag_' in col:
            base_feat = col.split('_lag_')[0]
            if base_feat in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(df_filled[base_feat].mean())
                continue
        
        if '_d14to' in col:
            # For forecast-safe rolling windows
            base_feat = col.split('_d14to')[0]
            if base_feat in df_filled.columns:
                df_filled[col] = df_filled[col].fillna(df_filled[base_feat].mean())
                continue
        
        # For other features, use standard fill methods
        if np.issubdtype(df_filled[col].dtype, np.number):
            # For numeric columns, use mean
            df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        else:
            # For categorical, use most frequent
            df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0] if len(df_filled[col].mode()) > 0 else None)
    
    # Check if we missed any
    remaining_missing = df_filled.isnull().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"Still have {remaining_missing} missing values after filling")
    else:
        logger.info("All missing values filled successfully")
    
    return df_filled

def engineer_weather_features(
    df: pd.DataFrame,
    date_col: str = 'calday'
) -> pd.DataFrame:
    """Complete weather feature engineering pipeline with forecast horizon constraints.
    
    This function creates lag features, rolling window features, seasonal features
    
    Args:
        df: Preprocessed weather DataFrame
        date_col: Name of the date column
        
    Returns:
        DataFrame with all engineered features
    """
    # Key features identified as most important
    key_temp_features = ['temp', 'tempmax', 'tempmin', 'feelslike', 'feelslikemax', 'feelslikemin']
    key_precip_features = ['precip', 'precipprob', 'precipcover', 'snow', 'snowdepth']
    key_atmo_features = ['humidity', 'windspeed', 'winddir', 'sealevelpressure', 'visibility']
    key_solar_features = ['solarradiation', 'solarenergy', 'cloudcover', 'daylight_hours']
    
    # All important features
    key_features = key_temp_features + key_precip_features + key_atmo_features + key_solar_features
    
    # 1. Create lag features (14+ days lags only)
    lag_days = [1, 2, 3, 7, 14, 28]
    df_features = create_lag_features(df, key_features, lag_days, date_col)
    
    # 2. Create forecast-safe rolling window features
    forecast_safe_windows = [(1, 3), (1, 7), (3, 14), (7, 14)]
    df_features = create_forecast_safe_rolling_features(df_features, key_features, forecast_safe_windows, date_col)
    
    # 3. Create seasonal features if multiple years available
    if date_col in df_features.columns:
        date_series = pd.to_datetime(df_features[date_col])
        if date_series.dt.year.nunique() > 1:
            df_features = create_seasonal_features(df_features, date_col)
    
    # 4. Create past extreme indicators
    df_features = create_past_extreme_indicators(df_features, key_features, [28, 42], date_col)
    
    # 5. Add domain-specific features
    df_features = create_temperature_features(df_features)
    
    # 6. Fill missing values
    df_features = fill_missing_values(df_features)
    
    logger.info(f"Completed feature engineering with forecast horizon constraints. Final shape: {df_features.shape}")
    return df_features
