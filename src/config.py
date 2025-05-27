"""Configuration settings for outlier detection and removal."""

# Data paths
DATA_DIR = "../data"
OUTPUT_DIR = "../data/processed"

# Outlier detection parameters
QNT_THRESHOLD = 10
ZSCORE_THRESHOLD = 10

# Essential columns for processing
ESSENTIAL_COLUMNS = ["index_material", "index_store", "format_merch", "geolocal_type", "qnt", "calday"]

# Processing parameters
CHUNK_SIZE = 500000
