import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd

from src.parquet_functions import write_parquet_file
from src.dims import update_lookup_table

DATA_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")
DIMS_DIR = Path("data/silver/dims")

SILVER_DIR.mkdir(parents=True, exist_ok=True)
DIMS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure pitch lookup table exists
pitch_lookup_path = DIMS_DIR / "pitch_lookup.parquet"
if not pitch_lookup_path.exists():
    print("Pitch lookup table not found. Creating it...")
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).parent / "make_pitch_lookup.py")], check=True)

# Load pitch lookup table
pitch_lookup = pd.read_parquet(pitch_lookup_path)

# Get all files to process
files = sorted(DATA_DIR.glob("*.parquet"))
total_files = len(files)
print(f"Processing {total_files} bronze files...")

for idx, file_path in enumerate(files, start=1):
    print(f"Processing file {idx}/{total_files} ({file_path.name})...", end='\r')

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"\nError reading {file_path.name}: {e}")
        print(f"Skipping file...")
        continue

    # Check if the file has the expected columns
    required_columns = ["game_pk", "ab_idx", "pitch_num_in_pa"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"\nWarning: {file_path.name} is missing columns: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")
        print(f"Skipping file...")
        continue

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

    # Sort by pitcher to group all pitches by the same pitcher together chronologically
    # This allows us to look back at previous pitches across at-bats
    df = df.sort_values(["game_pk", "pitcher_idx", "ab_idx", "pitch_num_in_pa"]).reset_index(drop=True)

    # Group by pitcher to create lag features across at-bats
    g = df.groupby(["game_pk", "pitcher_idx"], sort=False)

    #Create Previous Pitch Columns: P-1, P-2, P-3...
    #P-1
    df["pitch_1_idx"] = g["pitch_idx"].shift(1).fillna(0).astype("int8")
    df["pitch_1_bucket_idx"] = g["pitch_bucket_idx"].shift(1).fillna(0).astype("int8")
    #P-2
    df["pitch_2_idx"] = g["pitch_idx"].shift(2).fillna(0).astype("int8")
    df["pitch_2_bucket_idx"] = g["pitch_bucket_idx"].shift(2).fillna(0).astype("int8")
    #P-3
    df["pitch_3_idx"] = g["pitch_idx"].shift(3).fillna(0).astype("int8")
    df["pitch_3_bucket_idx"] = g["pitch_bucket_idx"].shift(3).fillna(0).astype("int8")
    #P-4
    df["pitch_4_idx"] = g["pitch_idx"].shift(4).fillna(0).astype("int8")
    df["pitch_4_bucket_idx"] = g["pitch_bucket_idx"].shift(4).fillna(0).astype("int8")
    #P-5
    df["pitch_5_idx"] = g["pitch_idx"].shift(5).fillna(0).astype("int8")
    df["pitch_5_bucket_idx"] = g["pitch_bucket_idx"].shift(5).fillna(0).astype("int8")
    #P-6
    df["pitch_6_idx"] = g["pitch_idx"].shift(6).fillna(0).astype("int8")
    df["pitch_6_bucket_idx"] = g["pitch_bucket_idx"].shift(6).fillna(0).astype("int8")
    #P-7
    df["pitch_7_idx"] = g["pitch_idx"].shift(7).fillna(0).astype("int8")
    df["pitch_7_bucket_idx"] = g["pitch_bucket_idx"].shift(7).fillna(0).astype("int8")
    #P-8
    df["pitch_8_idx"] = g["pitch_idx"].shift(8).fillna(0).astype("int8")
    df["pitch_8_bucket_idx"] = g["pitch_bucket_idx"].shift(8).fillna(0).astype("int8")
    #P-9
    df["pitch_9_idx"] = g["pitch_idx"].shift(9).fillna(0).astype("int8")
    df["pitch_9_bucket_idx"] = g["pitch_bucket_idx"].shift(9).fillna(0).astype("int8")
    #P-10
    df["pitch_10_idx"] = g["pitch_idx"].shift(10).fillna(0).astype("int8")
    df["pitch_10_bucket_idx"] = g["pitch_bucket_idx"].shift(10).fillna(0).astype("int8")

    # Sort back to original order (by ab_idx, pitch_num) for output
    df = df.sort_values(["game_pk", "ab_idx", "pitch_num_in_pa"]).reset_index(drop=True)


    # --- Fail loudly if mapping missed anything
    if df["pitcher_idx"].isna().any():
        missing_count = df["pitcher_idx"].isna().sum()
        raise RuntimeError(f"Missing pitcher_idx after merge for {file_path}. {missing_count} rows affected")
    if df["batter_idx"].isna().any():
        missing_count = df["batter_idx"].isna().sum()
        raise RuntimeError(f"Missing batter_idx after merge for {file_path}. {missing_count} rows affected")
    if df["pitch_idx"].isna().any():
        missing_count = df["pitch_idx"].isna().sum()
        raise RuntimeError(f"Missing pitch_idx after merge for {file_path}. {missing_count} rows affected")
    if df["pitch_bucket_idx"].isna().any():
        missing_count = df["pitch_bucket_idx"].isna().sum()
        raise RuntimeError(f"Missing pitch_bucket_idx after merge for {file_path}. {missing_count} rows affected")


    # --- Determine output name from input filename
    # Input format: {game_date}_game_{game_pk}.parquet
    # Output format: {game_date}_game_{game_pk}.parquet (same)
    out_path = SILVER_DIR / file_path.name

    write_parquet_file(df, out_path)

    if idx % 10 == 0 or idx == total_files:
        print(f"Processed {idx}/{total_files} files ({idx/total_files*100:.1f}%)")

print("\nProcessing complete!")

    #     Column                Dtype
    # --  ------                -----
    # 0   game_date             object
    # 1   game_pk               int64
    # 2   ab_idx                int64
    # 3   pitch_num_in_pa       int64
    # 4   inning                int64
    # 5   pitcher_is_home       bool
    # 6   pitcher_is_right      bool
    # 7   batter_is_right       bool
    # 8   pitcher_idx           int32
    # 9   batter_idx            int32
    # 10  balls_before_pitch    int8
    # 11  strikes_before_pitch  int8
    # 12  pitch_idx             int64
    # 13  pitch_bucket_idx      int64
    # 14  pitch_1_idx           int8
    # 15  pitch_1_bucket_idx    int8
    # 16  pitch_2_idx           int8
    # 17  pitch_2_bucket_idx    int8
    # 18  pitch_3_idx           int8
    # 19  pitch_3_bucket_idx    int8
    # 20  pitch_4_idx           int8
    # 21  pitch_4_bucket_idx    int8
    # 22  pitch_5_idx           int8
    # 23  pitch_5_bucket_idx    int8
    # 24  pitch_6_idx           int8
    # 25  pitch_6_bucket_idx    int8
    # 26  pitch_7_idx           int8
    # 27  pitch_7_bucket_idx    int8
    # 28  pitch_8_idx           int8
    # 29  pitch_8_bucket_idx    int8
    # 30  pitch_9_idx           int8
    # 31  pitch_9_bucket_idx    int8
    # 32  pitch_10_idx          int8
    # 33  pitch_10_bucket_idx   int8  