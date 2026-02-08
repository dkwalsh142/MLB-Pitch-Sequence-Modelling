import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import time

from src.mlb_api import get_game_pks
from src.mlb_api import get_pitch_feed
from src.parquet_functions import write_parquet_file

# Split into season chunks to avoid API limits on schedule endpoint
date_ranges = [
    ("2023-03-27", "2023-10-31"),  # 2023 season
    ("2024-03-20", "2024-10-31"),  # 2024 season
    ("2025-03-27", "2025-09-28"),  # 2025 season
]

game_pks = []
for start, end in date_ranges:
    season_pks = get_game_pks(start, end)
    print(f"Found {len(season_pks)} games for {start} to {end}")
    game_pks.extend(season_pks)

# Remove duplicates (in case any games span seasons)
game_pks = list(set(game_pks))

total_games = len(game_pks)
print(f"Found {total_games} total games to download")

# Check which games already exist
bronze_dir = Path("data/bronze")
bronze_dir.mkdir(parents=True, exist_ok=True)
existing_files = set(bronze_dir.glob("*.parquet"))
existing_game_pks = set()
for file in existing_files:
    # Extract game_pk from filename pattern: {date}_game_{game_pk}.parquet
    try:
        game_pk = int(file.stem.split("_game_")[-1])
        existing_game_pks.add(game_pk)
    except (ValueError, IndexError):
        pass

games_to_download = [pk for pk in game_pks if pk not in existing_game_pks]
skipped_count = len(game_pks) - len(games_to_download)

if skipped_count > 0:
    print(f"Skipping {skipped_count} already downloaded games")
print(f"Downloading {len(games_to_download)} games")

for idx, game_pk in enumerate(games_to_download, start=1):
    print(f"Downloading game {idx}/{len(games_to_download)} (game_pk: {game_pk})...", end='\r')
    df, game_date = get_pitch_feed(game_pk)
    write_parquet_file(
        df,
        f"data/bronze/{game_date}_game_{game_pk}.parquet"
        )
    if idx % 10 == 0 or idx == len(games_to_download):
        print(f"Downloaded {idx}/{len(games_to_download)} games ({idx/len(games_to_download)*100:.1f}%)")

    # Rate limiting: sleep between requests (except after the last one)
    if idx < len(games_to_download):
        time.sleep(0.6)

print("\nDownload complete!")


#      Column               Dtype
# ---  ------               -----
#  0   game_date            object
#  1   game_pk              int64
#  2   ab_idx               int64
#  3   pitch_num_in_pa      int64
#  4   inning               int64
#  5   pitcher_id           int64
#  6   batter_id            int64
#  7   pitcher_hand         object
#  8   batter_hand          object
#  9   pitcher_is_home      bool
#  10  balls_after_pitch    int64
#  11  strikes_after_pitch  int64
#  12  pitch_type           object




