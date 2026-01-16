import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.parquet_functions import write_parquet_file
from src.dims import update_lookup_table

DATA_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")
DIMS_DIR = Path("data/silver/dims")

SILVER_DIR.mkdir(parents=True, exist_ok=True)
DIMS_DIR.mkdir(parents=True, exist_ok=True)

# Load pitch lookup table
pitch_lookup = pd.read_parquet(DIMS_DIR / "pitch_lookup.parquet")

for file_path in sorted(DATA_DIR.iterdir()):
    if not file_path.is_file():
        continue
    if file_path.suffix != ".parquet":
        continue
    df = pd.read_parquet(file_path)

    #make sure pitches are in order
    df = df.sort_values(["game_pk", "ab_idx", "pitch_num_in_pa"]).reset_index(drop=True)

    # --- Pitcher and batter hands
    df["pitcher_is_right"] = df["pitcher_hand"] == "R"
    df["batter_is_right"] = df["batter_hand"].isin(["R", "S"])
    
    # --- Drop raw handedness columns
    df = df.drop(columns=["pitcher_hand", "batter_hand"])

    # --- Pithcer and Batter Idnexes
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

    # --- Drop redundant columns after pitcher/batter lookup merge
    df = df.drop(columns=["pitcher_id","batter_id"])

    g = df.groupby(["game_pk", "ab_idx"], sort=False)

    # Pre-pitch count from prior pitch's post-pitch count
    df["balls_before_pitch"] = g["balls_after_pitch"].shift(1).fillna(0).astype("int8")
    df["strikes_before_pitch"] = g["strikes_after_pitch"].shift(1).fillna(0).astype("int8")

    # Drop balls and strikes after pitch
    df = df.drop(columns=["balls_after_pitch","strikes_after_pitch"])

    # --- Map unknown pitch types to a default
    # Check for unknown pitch codes and map them to "UN" (unknown)
    known_pitch_codes = set(pitch_lookup["pitch_code"])
    df["pitch_type"] = df["pitch_type"].apply(
        lambda x: x if x in known_pitch_codes else "UN"
    )

    df = df.merge(pitch_lookup, left_on="pitch_type", right_on="pitch_code", how="left")

    # --- Drop redundant columns after pitch lookup merge
    df = df.drop(columns=["pitch_type", "pitch_code", "pitch_bucket"])

    # Recreate groupby object after adding pitch columns
    g = df.groupby(["game_pk", "ab_idx"], sort=False)

    #Create Previuous Pithc Collumns: P-1, P-2, P-3
    #P-1
    df["pitch-1_idx"] = g["pitch_idx"].shift(1).fillna(0).astype("int8")
    df["pitch-1_bucket_idx"] = g["pitch_bucket_idx"].shift(1).fillna(0).astype("int8")
    #P-2
    df["pitch-2_idx"] = g["pitch_idx"].shift(2).fillna(0).astype("int8")
    df["pitch-2_bucket_idx"] = g["pitch_bucket_idx"].shift(2).fillna(0).astype("int8")
    #P-3
    df["pitch-3_idx"] = g["pitch_idx"].shift(3).fillna(0).astype("int8")
    df["pitch-3_bucket_idx"] = g["pitch_bucket_idx"].shift(3).fillna(0).astype("int8")


    # --- Fail loudly if mapping missed anything
    if df["pitcher_idx"].isna().any():
        missing_ids = df.loc[df["pitcher_idx"].isna(), "pitcher_id"].unique()
        raise RuntimeError(f"Missing pitcher_idx after merge for {file_path}. Missing pitcher_id(s): {missing_ids.tolist()}")
    if df["batter_idx"].isna().any():
        missing_ids = df.loc[df["batter_idx"].isna(), "batter_id"].unique()
        raise RuntimeError(f"Missing batter_idx after merge for {file_path}. Missing batter_id(s): {missing_ids.tolist()}")
    if df["pitch_idx"].isna().any():
        # Note: pitch_type is already dropped at this point, so we can't show which types are missing
        missing_count = df["pitch_idx"].isna().sum()
        raise RuntimeError(f"Missing pitch_idx after merge for {file_path}. {missing_count} rows affected")
    if df["pitch_bucket_idx"].isna().any():
        missing_count = df["pitch_bucket_idx"].isna().sum()
        raise RuntimeError(f"Missing pitch_bucket_idx after merge for {file_path}. {missing_count} rows affected")


    # --- Determine output name
    game_pk = int(df["game_pk"].iloc[0]) if "game_pk" in df.columns else file_path.stem
    out_path = SILVER_DIR / f"game_{game_pk}.parquet"

    
    write_parquet_file(df, out_path)

    #     Column                Dtype 
    # --  ------                ----- 
    # 0   game_pk               object
    # 1   ab_idx                int64 
    # 2   pitch_num_in_pa       int64 
    # 3   inning                int64 
    # 4   pitcher_is_home       bool  
    # 5   pitcher_is_right      bool  
    # 6   batter_is_right       bool  
    # 7   pitcher_idx           int32 
    # 8   batter_idx            int32 
    # 9   balls_before_pitch    int8  
    # 10  strikes_before_pitch  int8  
    # 11  pitch_idx             int64 
    # 12  pitch_bucket_idx      int64 
    # 13  pitch-1_idx           int8  
    # 14  pitch-1_bucket_idx    int8  
    # 15  pitch-2_idx           int8  
    # 16  pitch-2_bucket_idx    int8  
    # 17  pitch-3_idx           int8  
    # 18  pitch-3_bucket_idx    int8  