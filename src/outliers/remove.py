"""Functions for removing outliers from sales data files."""

import os
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

from src.config import DATA_DIR, OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_outliers(outliers: pd.DataFrame, output_path: str = None) -> None:
    """Save detected outliers to CSV.
    
    Args:
        outliers: DataFrame containing outlier records
        output_path: Path to save the outliers file (default: DATA_DIR/outliers.csv)
    """
    if outliers.empty:
        logger.warning("No outliers to save")
        return
    
    if output_path is None:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, 'outliers.csv')
    
    # Save outliers
    outliers.to_csv(output_path, index=False)
    logger.info(f"Saved {len(outliers)} outliers to {output_path}")


def remove_outliers_from_file(
    file_path: str, 
    outliers: pd.DataFrame, 
    output_dir: str = None
) -> Tuple[str, int]:
    """Remove outliers from a data file.
    
    Args:
        file_path: Path to the data file
        outliers: DataFrame containing outliers to remove
        output_dir: Directory to save the cleaned file (default: OUTPUT_DIR)
        
    Returns:
        Tuple of (cleaned_file_path, removed_count)
    """
    if outliers.empty:
        logger.info(f"No outliers to remove from {os.path.basename(file_path)}")
        return (file_path, 0)
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Extract month from filename to filter outliers
        month = os.path.basename(file_path).replace('.csv', '').replace('2023', '')
        
        # Filter outliers by month to reduce processing time
        month_outliers = outliers
        if 'source_month' in outliers.columns:
            month_outliers = outliers[outliers['source_month'] == month].copy()
        
        if month_outliers.empty:
            logger.info(f"No outliers for month {month}, skipping {os.path.basename(file_path)}")
            return (file_path, 0)
        
        # Determine separator
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
            sep = ';' if ';' in first_line else ','
        
        # Load data in chunks
        logger.info(f"Processing {os.path.basename(file_path)}...")
        chunks = []
        chunk_size = 100000  # Process in reasonable chunk sizes
        outliers_removed = 0
        
        # Prepare output file
        base_name = os.path.basename(file_path)
        name_parts = os.path.splitext(base_name)
        cleaned_file = os.path.join(output_dir, f"{name_parts[0]}_clean{name_parts[1]}")
        
        # Prepare for matching outliers
        if 'calday' in month_outliers.columns:
            # Handle different calday formats
            if pd.api.types.is_datetime64_any_dtype(month_outliers['calday']):
                # Convert datetime to integer format YYYYMMDD for matching
                month_outliers['calday_int'] = month_outliers['calday'].dt.strftime('%Y%m%d').astype(int)
            else:
                try:
                    # If it's already a string or int, ensure it's in integer format
                    month_outliers['calday_int'] = month_outliers['calday'].astype(int)
                except:
                    # If it's another string format, parse as datetime first
                    month_outliers['calday_int'] = pd.to_datetime(month_outliers['calday']).dt.strftime('%Y%m%d').astype(int)
        
        # Create match keys
        outlier_keys = set()
        for _, row in month_outliers.iterrows():
            store = int(row['index_store'])
            material = int(row['index_material'])
            day = int(row['calday_int'] if 'calday_int' in month_outliers.columns else row['calday'])
            outlier_keys.add((store, material, day))
        
        # Process file in chunks
        reader = pd.read_csv(file_path, sep=sep, chunksize=chunk_size)
        
        # Process first chunk
        for i, chunk in enumerate(tqdm(reader, desc=f"Processing {os.path.basename(file_path)}")):
            # Track whether this is the first chunk for header writing
            first_chunk = (i == 0)
            
            # Create record keys for matching
            keys_to_check = []
            for idx, row in chunk.iterrows():
                if pd.notna(row['index_store']) and pd.notna(row['index_material']) and pd.notna(row['calday']):
                    store = int(row['index_store'])
                    material = int(row['index_material'])
                    day = int(row['calday'])
                    keys_to_check.append(((store, material, day), idx))
            
            # Identify records to remove
            indices_to_remove = []
            for key, idx in keys_to_check:
                if key in outlier_keys:
                    indices_to_remove.append(idx)
            
            # Remove outlier records
            if indices_to_remove:
                chunk = chunk.drop(indices_to_remove)
                outliers_removed += len(indices_to_remove)
            
            # Save chunk to file (append mode after first chunk)
            if first_chunk:
                chunk.to_csv(cleaned_file, sep=sep, index=False)
            else:
                chunk.to_csv(cleaned_file, sep=sep, index=False, mode='a', header=False)
        
        logger.info(f"Removed {outliers_removed} outliers from {os.path.basename(file_path)}")
        logger.info(f"Saved cleaned file to {cleaned_file}")
        
        return (cleaned_file, outliers_removed)
        
    except Exception as e:
        logger.error(f"Error cleaning {file_path}: {str(e)}")
        return (file_path, 0)


def clean_all_files(
    data_dir: str,
    outliers: pd.DataFrame,
    months: List[int] = list(range(1, 13)),
    year: int = 2023,
    output_dir: str = None
) -> Dict[str, int]:
    """Clean all monthly data files by removing outliers.
    
    Args:
        data_dir: Directory containing data files
        outliers: DataFrame containing outliers to remove
        months: List of months to process
        year: Year of data files
        output_dir: Directory to save cleaned files
        
    Returns:
        Dictionary mapping file names to counts of removed records
    """
    if outliers.empty:
        logger.warning("No outliers provided for removal")
        return {}
    
    # Find monthly data files
    all_files = [os.path.join(data_dir, f"{year}{month:02d}.csv") for month in months]
    existing_files = [f for f in all_files if os.path.exists(f)]
    
    if not existing_files:
        logger.warning(f"No data files found in {data_dir} matching the pattern {year}MM.csv")
        return {}
    
    logger.info(f"Found {len(existing_files)} data files to clean")
    
    # Process each file
    results = {}
    for file_path in existing_files:
        base_name = os.path.basename(file_path)
        cleaned_path, removed_count = remove_outliers_from_file(file_path, outliers, output_dir)
        results[base_name] = removed_count
    
    # Summarize results
    total_removed = sum(results.values())
    logger.info(f"Total outliers removed across all files: {total_removed}")
    
    return results
