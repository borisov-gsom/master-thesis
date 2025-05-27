"""Functions for preprocessing weather data."""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_weather_data(file_path: str) -> pd.DataFrame:
    """Load weather data from CSV file.
    
    Args:
        file_path: Path to the weather data CSV file
        
    Returns:
        DataFrame containing weather data
    """
    logger.info(f"Loading weather data from {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"Weather data file not found: {file_path}")
        raise FileNotFoundError(f"Weather data file not found: {file_path}")
    
    # Load data
    df = pd.read_csv(file_path, sep=',')
    
    logger.info(f"Loaded weather data: {len(df)} rows, {len(df.columns)} columns")
    return df

def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean weather data by removing unnecessary columns and converting types.
    
    Args:
        df: DataFrame containing weather data
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning weather data")
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Remove unnecessary columns
    columns_to_remove = ['name', 'stations', 'icon', 'description', 'severerisk']
    
    # Check that these columns exist before removing
    columns_to_remove = [col for col in columns_to_remove if col in df_clean.columns]
    
    # Remove columns
    df_clean = df_clean.drop(columns=columns_to_remove)
    
    # Convert datetime column to proper datetime format
    df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])
    
    # Set datetime as index for time series analysis
    df_clean = df_clean.set_index('datetime').sort_index()
    
    # Process sunrise/sunset times to calculate daylight duration
    if 'sunrise' in df_clean.columns and 'sunset' in df_clean.columns:
        df_clean['sunrise'] = pd.to_datetime(df_clean['sunrise'])
        df_clean['sunset'] = pd.to_datetime(df_clean['sunset'])
        
        # Calculate daylight duration in hours
        df_clean['daylight_hours'] = (df_clean['sunset'] - df_clean['sunrise']).dt.total_seconds() / 3600
        
        # Remove the original sunrise/sunset columns
        df_clean = df_clean.drop(columns=['sunrise', 'sunset'])
    
    # Handle missing values in preciptype
    if 'preciptype' in df_clean.columns:
        # Create binary flags for precipitation types
        df_clean['has_precipitation'] = df_clean['preciptype'].notna()
        df_clean['has_rain'] = df_clean['preciptype'].str.contains('rain', na=False)
        df_clean['has_snow'] = df_clean['preciptype'].str.contains('snow', na=False)
        
        # Drop the original preciptype column
        df_clean = df_clean.drop(columns=['preciptype'])
    
    # Process conditions column if it exists
    if 'conditions' in df_clean.columns:
        # Common weather conditions to extract
        weather_types = ['Clear', 'Overcast', 'Rain', 'Snow', 'Fog', 'Cloudy', 'Partially cloudy']
        
        for weather_type in weather_types:
            col_name = f'is_{weather_type.lower().replace(" ", "_")}'
            df_clean[col_name] = df_clean['conditions'].str.contains(weather_type, case=False, na=False)
        
        # Drop the original conditions column
        df_clean = df_clean.drop(columns=['conditions'])
    
    logger.info(f"Cleaned weather data: {df_clean.shape}")
    return df_clean

def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract date-related features from the datetime index.
    
    Args:
        df: DataFrame with datetime index
        
    Returns:
        DataFrame with added date features
    """
    logger.info("Extracting date features")
    
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # Extract basic date components
    df_features['year'] = df_features.index.year
    df_features['month'] = df_features.index.month
    df_features['day'] = df_features.index.day
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['day_of_year'] = df_features.index.dayofyear
    df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6])  # 5=Saturday, 6=Sunday
    
    # Add quarter and seasons
    df_features['quarter'] = df_features.index.quarter
    df_features['season'] = df_features.index.month.map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'fall', 10: 'fall', 11: 'fall'
    })
    
    # Add sin/cos features for cyclical month and day of year
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['day_of_year_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365)
    df_features['day_of_year_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365)
    
    logger.info(f"Extracted date features: {df_features.shape}")
    return df_features

def prepare_weather_data(file_path: str, reset_index: bool = True) -> pd.DataFrame:
    """Complete weather data preprocessing pipeline.
    
    Args:
        file_path: Path to the weather data file
        reset_index: Whether to reset the datetime index to a column (default: True)
        
    Returns:
        Preprocessed weather DataFrame
    """
    # Load data
    df = load_weather_data(file_path)
    
    # Clean data
    df_clean = clean_weather_data(df)
    
    # Extract date features
    df_features = extract_date_features(df_clean)
    
    # Reset index if requested
    if reset_index:
        df_features = df_features.reset_index()
        # Rename datetime to calday for consistency with other datasets
        df_features = df_features.rename(columns={'datetime': 'calday'})
    
    logger.info("Weather data preprocessing completed")
    return df_features
