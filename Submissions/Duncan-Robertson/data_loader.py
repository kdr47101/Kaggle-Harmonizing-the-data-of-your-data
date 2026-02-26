import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Define paths relative to this file
# Assumes this file is in Submissions/Duncan-Robertson/
# and data is in data/ at the project root (2 levels up)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'

TRAIN_TEXT_DIR = DATA_DIR / 'TrainingPubText'
TEST_TEXT_DIR = DATA_DIR / 'TestPubText'
TRAIN_SDRF_DIR = DATA_DIR / 'TrainingSDRFs'

def _flatten_json(data: Any) -> str:
    """
    Recursively extracts text from a JSON object (dict or list).
    Joins strings with newlines to create a single text document.
    """
    if isinstance(data, dict):
        # Extract text from all values in the dictionary
        # We use a double newline to separate potential sections
        return "\n\n".join(_flatten_json(v) for v in data.values() if v)
    elif isinstance(data, list):
        # Extract text from all items in the list
        return "\n\n".join(_flatten_json(v) for v in data if v)
    elif isinstance(data, str):
        return data.strip()
    return ""

def load_train_texts() -> Dict[str, str]:
    """
    Loads training manuscript texts from JSON files.
    
    Returns:
        Dict[str, str]: Dictionary mapping paper ID (filename without extension) 
                        to the full flattened manuscript text.
    """
    texts = {}
    if not TRAIN_TEXT_DIR.exists():
        print(f"Warning: {TRAIN_TEXT_DIR} does not exist.")
        return texts
        
    print(f"Loading training texts from {TRAIN_TEXT_DIR}...")
    for file_path in TRAIN_TEXT_DIR.glob('*.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            texts[file_path.stem] = _flatten_json(data)
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            
    return texts

def load_test_texts() -> Dict[str, str]:
    """
    Loads test manuscript texts from JSON files.
    
    Returns:
        Dict[str, str]: Dictionary mapping paper ID (filename without extension) 
                        to the full flattened manuscript text.
    """
    texts = {}
    if not TEST_TEXT_DIR.exists():
        print(f"Warning: {TEST_TEXT_DIR} does not exist.")
        return texts
        
    print(f"Loading test texts from {TEST_TEXT_DIR}...")
    for file_path in TEST_TEXT_DIR.glob('*.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            texts[file_path.stem] = _flatten_json(data)
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            
    return texts

def load_train_sdrfs() -> Dict[str, pd.DataFrame]:
    """
    Loads training SDRF files.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping paper ID to the SDRF DataFrame.
    """
    sdrfs = {}
    if not TRAIN_SDRF_DIR.exists():
        print(f"Warning: {TRAIN_SDRF_DIR} does not exist.")
        return sdrfs
        
    print(f"Loading training SDRFs from {TRAIN_SDRF_DIR}...")
    for file_path in TRAIN_SDRF_DIR.glob('*'):
        # Check for common SDRF extensions
        if file_path.suffix not in ['.tsv', '.csv', '.txt']:
            continue
            
        try:
            # SDRF is typically tab-delimited, but we handle CSV just in case
            sep = '\t' if file_path.suffix in ['.tsv', '.txt'] else ','
            df = pd.read_csv(file_path, sep=sep)
            sdrfs[file_path.stem] = df
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            
    return sdrfs

def load_papers(split: str = 'test') -> Dict[str, str]:
    """
    Wrapper to load manuscript texts for the pipeline.
    
    Args:
        split (str): 'train' or 'test'. Defaults to 'test'.
        
    Returns:
        Dict[str, str]: Dictionary mapping paper ID to full text.
    """
    if split == 'train':
        return load_train_texts()
    return load_test_texts()

if __name__ == "__main__":
    # Basic testing of the loader functions
    train_texts = load_train_texts()
    print(f"Loaded {len(train_texts)} training texts.")
    
    sdrfs = load_train_sdrfs()
    print(f"Loaded {len(sdrfs)} training SDRFs.")
    
    test_texts = load_test_texts()
    print(f"Loaded {len(test_texts)} test texts.")
    
    # Verify overlap between training texts and SDRFs
    common_ids = set(train_texts.keys()) & set(sdrfs.keys())
    print(f"Training samples with both text and SDRF: {len(common_ids)}")