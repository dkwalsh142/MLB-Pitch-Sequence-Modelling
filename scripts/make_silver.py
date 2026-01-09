import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.parquet_functions import write_parquet_file
from src.dims import update_lookup_table



DATA_DIR = Path("data/bronze")

pitcher_lookup = {}
batter_lookup = {}


DATA_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")
DIMS_DIR = Path("data/silver/dims")

SILVER_DIR.mkdir(parents=True, exist_ok=True)
DIMS_DIR.mkdir(parents=True, exist_ok=True)

for file_path in sorted(DATA_DIR.iterdir()):
    if not file_path.is_file():
        continue
    if file_path.suffix != ".parquet":
        continue
    df = pd.read_parquet(file_path)

    # --- Derived columns
    df["pitcher_is_right"] = df["pitcher_hand"] == "R"
    df["batter_is_right"] = df["batter_hand"].isin(["R", "S"])
    
    # --- Drop raw handedness columns
    df = df.drop(columns=["pitcher_hand", "batter_hand"])

    # --- Update / load dims (append-only)
    pitcher_lookup = update_lookup_table(
    DIMS_DIR / "pitcher_lookup.parquet",
    "pitcher_id",
    "pitcher_idx",
    df["pitcher_id"],
    )
    batter_lookup = update_lookup_table(
    DIMS_DIR / "batter_lookup.parquet",
    "batter_id",
    "batter_idx",
    df["batter_id"],
    )
    
    # --- Attach indices
    df = df.merge(pitcher_lookup, on="pitcher_id", how="left")
    df = df.merge(batter_lookup,  on="batter_id",  how="left")

    # --- Fail loudly if mapping missed anything
    if df["pitcher_idx"].isna().any():
        raise RuntimeError(f"Missing pitcher_idx after merge for {file_path}")
    if df["batter_idx"].isna().any():
        raise RuntimeError(f"Missing batter_idx after merge for {file_path}")


    # --- Determine output name
    game_pk = int(df["game_pk"].iloc[0]) if "game_pk" in df.columns else file_path.stem
    out_path = SILVER_DIR / f"game_{game_pk}.parquet"

    # If your helper takes (df, path), prefer passing a Path:
    write_parquet_file(df, out_path)