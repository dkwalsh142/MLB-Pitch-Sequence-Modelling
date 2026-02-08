import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import shutil

DATA_DIR = Path("data/silver")
TRAIN_DEST = Path("data/gold/train")
VAL_DEST = Path("data/gold/val")
TEST_DEST = Path("data/gold/test")

TRAIN_DEST.mkdir(parents=True, exist_ok=True)
VAL_DEST.mkdir(parents=True, exist_ok=True)
TEST_DEST.mkdir(parents=True, exist_ok=True)

# Get all parquet files sorted by filename (which includes date)
parquet_files = sorted(DATA_DIR.glob("*.parquet"))
num_files = len(parquet_files)

TRAIN_SPLIT = .8
VAL_SPLIT = .1
TEST_SPLIT = .1

final_train_index = TRAIN_SPLIT * num_files
final_val_index = final_train_index + VAL_SPLIT * num_files

for i, file_path in enumerate(parquet_files):
    if i < final_train_index:
        dest = TRAIN_DEST / file_path.name
        shutil.copy2(file_path, dest)
    elif i < final_val_index:
        dest = VAL_DEST / file_path.name
        shutil.copy2(file_path, dest)
    else:
        dest = TEST_DEST / file_path.name
        shutil.copy2(file_path, dest)