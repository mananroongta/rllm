# Standard imports
import os

# Base directory setup
# Default to "examples/finqa/data" relative to the repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "examples", "finqa", "data")

# Allow overriding via environment variable
BASE_DATA_DIR = os.getenv("FINQA_DATA_DIR", DEFAULT_DATA_DIR)

# Data subdirectories
MULTI_TABLE_DATA_DIR = os.path.join(BASE_DATA_DIR, "multi_table_data")
BENCHMARK_DATA_DIR = os.path.join(BASE_DATA_DIR, "test_benchmark")

# Table storage
TABLES_ROOT = os.getenv("FINQA_TABLES_ROOT", os.path.join(BASE_DATA_DIR, "company_tables"))
TABLES_CLEANED_ALL_COMPANIES_FILE_NAME = "tables_cleaned_all_companies.json"

# Generated files
GENERATED_DATA_DIR = os.path.join(BASE_DATA_DIR, "generated")

# Task Data Paths
TRAIN_QUESTIONS_PATH = os.path.join(BASE_DATA_DIR, "train_finqa.csv")
VAL_QUESTIONS_PATH = os.path.join(BASE_DATA_DIR, "val_finqa.csv")
TEST_QUESTIONS_PATH = os.path.join(BASE_DATA_DIR, "test_finqa.csv")

MULTI_TABLE_TRAIN_PATH = os.path.join(MULTI_TABLE_DATA_DIR, "train_finqa.csv")
MULTI_TABLE_VAL_PATH = os.path.join(MULTI_TABLE_DATA_DIR, "val_finqa.csv")
MULTI_TABLE_TEST_PATH = os.path.join(MULTI_TABLE_DATA_DIR, "test_finqa.csv")
