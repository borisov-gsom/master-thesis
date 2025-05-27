"""Functions for loading sales data efficiently."""

import os
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

from src.config import ESSENTIAL_COLUMNS, CHUNK_SIZE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_separator(file_path: str) -> str:
    """Detect whether a CSV file uses comma or semicolon as separator.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Detected separator (';' or ',')
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip()
        return ';' if ';' in first_line else ','


def load_monthly_data(
    data_path: str, 
    months: List[int] = list(range(1, 13)),
    year: int = 2023,
    columns: Optional[List[str]] = None,
    batch_size: int = 2
) -> pd.DataFrame:
    """Load monthly sales data efficiently.
    
    Args:
        data_path: Directory containing the data files
        months: List of months to load (1-12)
        year: Year of the data
        columns: Columns to load (None for all)
        batch_size: Number of files to load at once
        
    Returns:
        DataFrame containing combined data
    """
    if columns is None:
        columns = ESSENTIAL_COLUMNS
    
    # Find available data files
    all_files = [os.path.join(data_path, f"{year}{month:02d}.csv") for month in months]
    existing_files = [f for f in all_files if os.path.exists(f)]
    
    logger.info(f"Found {len(existing_files)} data files out of {len(all_files)} expected files")
    
    if not existing_files:
        logger.warning("No data files found!")
        return pd.DataFrame()
    
    # Process files in batches for memory efficiency
    df_combined = pd.DataFrame()
    
    for i in range(0, len(existing_files), batch_size):
        batch_files = existing_files[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}: {[os.path.basename(f) for f in batch_files]}")
        
        batch_dfs = []
        for file in tqdm(batch_files, desc="Loading files"):
            try:
                # Determine separator
                sep = detect_separator(file)
                
                # Check which columns actually exist in the file
                all_cols = pd.read_csv(file, sep=sep, nrows=1).columns.tolist()
                usable_columns = [col for col in columns if col in all_cols]
                
                # Read file in chunks
                chunks = []
                for chunk in pd.read_csv(file, sep=sep, usecols=usable_columns, chunksize=CHUNK_SIZE):
                    # Add month info
                    month = os.path.basename(file).replace('.csv', '').replace(str(year), '')
                    chunk['source_month'] = month
                    
                    # Ensure calday is in datetime format
                    if 'calday' in chunk.columns:
                        if chunk['calday'].dtype == 'object':
                            try:
                                chunk['calday'] = pd.to_datetime(chunk['calday'])
                            except:
                                try:
                                    chunk['calday'] = pd.to_datetime(chunk['calday'], format='%Y%m%d')
                                except:
                                    logger.warning(f"Could not convert calday to datetime in {file}")
                        elif not pd.api.types.is_datetime64_any_dtype(chunk['calday']):
                            try:
                                chunk['calday'] = pd.to_datetime(chunk['calday'].astype(str), format='%Y%m%d')
                            except:
                                logger.warning(f"Could not convert calday to datetime in {file}")
                    
                    chunks.append(chunk)
                
                # Combine chunks
                if chunks:
                    file_df = pd.concat(chunks, ignore_index=True)
                    batch_dfs.append(file_df)
                    logger.info(f"Loaded {os.path.basename(file)} with {len(file_df)} rows")
                    
                    # Clean up
                    del chunks, file_df
                    import gc
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
        
        # Combine batch dataframes
        if batch_dfs:
            batch_df = pd.concat(batch_dfs, ignore_index=True)
            if df_combined.empty:
                df_combined = batch_df
            else:
                df_combined = pd.concat([df_combined, batch_df], ignore_index=True)
            
            # Clean up
            del batch_dfs, batch_df
            import gc
            gc.collect()
    
    # Fill missing values in qnt
    if 'qnt' in df_combined.columns:
        df_combined['qnt'] = df_combined['qnt'].fillna(0)
    
    if not df_combined.empty:
        logger.info(f"Combined dataset has {len(df_combined)} rows and {len(df_combined.columns)} columns")
    else:
        logger.warning("No data was loaded. Please check file paths and formats.")
    
    return df_combined
