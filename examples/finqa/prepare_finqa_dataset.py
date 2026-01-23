import json
import os
import sys
import pandas as pd

# Allow running from examples/finqa or root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from rllm.data.dataset import DatasetRegistry
# Import constants to get path logic
from rllm.tools import fin_qa_constants as C

def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}. Ensure FINQA_DATA_DIR is set correctly or data is migrated.")
    return pd.read_csv(path)

def _parse_json_list(value):
    """Decode columns stored as JSON strings, defaulting to [] for empty values."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = stripped
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, str):
            cleaned = parsed.strip()
            return [cleaned] if cleaned else []
        return []
    return []


def prepare_finqa_data():
    print(f"Loading data from: {C.BASE_DATA_DIR}")
    
    single_train = _load_csv(C.TRAIN_QUESTIONS_PATH)
    single_val = _load_csv(C.VAL_QUESTIONS_PATH)
    single_test = _load_csv(C.TEST_QUESTIONS_PATH)

    multi_train = _load_csv(C.MULTI_TABLE_TRAIN_PATH)
    multi_val = _load_csv(C.MULTI_TABLE_VAL_PATH)
    multi_test = _load_csv(C.MULTI_TABLE_TEST_PATH)

    # Dataset Construction Strategy (Matches original)
    merged_train = pd.concat([single_train, single_val], ignore_index=True)
    merged_val = pd.concat([single_val], ignore_index=True)
    merged_test = pd.concat([single_test], ignore_index=True).sample(n=min(250, len(single_test)), random_state=42)

    def preprocess_fn(example):
        return {
            "question": example["user_query"],
            "ground_truth": example["answer"],
            "data_source": "finqa",
            "company": example["company"],
            "question_id": str(example["id"]),
            "question_type": example["question_type"],
            "core_question": example["question"],
            "table_name": _parse_json_list(example.get("table_name")),
            "columns_used": _parse_json_list(example.get("columns_used_json")),
            "rows_used": _parse_json_list(example.get("rows_used_json")),
            "explanation": example["explanation"],
        }

    train_processed = [preprocess_fn(row) for _, row in merged_train.iterrows()]
    val_processed = [preprocess_fn(row) for _, row in merged_val.iterrows()]
    test_processed = [preprocess_fn(row) for _, row in merged_test.iterrows()]

    train_dataset = DatasetRegistry.register_dataset("finqa", train_processed, "train")
    val_dataset = DatasetRegistry.register_dataset("finqa", val_processed, "val")
    test_dataset = DatasetRegistry.register_dataset("finqa", test_processed, "test")
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    try:
        train_dataset, val_dataset, test_dataset = prepare_finqa_data()
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        print(f"Train dataset path: {train_dataset.get_data_path()}")
        print(f"Validation dataset path: {val_dataset.get_data_path()}")
        print(f"Test dataset path: {test_dataset.get_data_path()}")
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        sys.exit(1)
