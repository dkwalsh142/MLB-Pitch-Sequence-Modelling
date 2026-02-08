"""
Quick script to check the date range of games in the bronze layer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from pathlib import Path

BRONZE_DIR = Path("data/bronze")

print("Checking bronze layer data...")
parquet_files = sorted(BRONZE_DIR.glob("*.parquet"))

if not parquet_files:
    print(f"No parquet files found in {BRONZE_DIR}")
    exit(1)

print(f"Found {len(parquet_files)} files")

# Extract all game_pks
print("\nExtracting game PKs...")
game_pks = [int(f.stem.replace("game_", "")) for f in parquet_files]

# Fetch dates for all games
print(f"Fetching dates for all {len(game_pks)} games from MLB API...")

def get_game_date(game_pk):
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"gamePk": game_pk}

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    dates = data.get("dates", [])
    if not dates:
        return None

    return dates[0]["date"]

game_dates = {}
for i, pk in enumerate(game_pks):
    if (i + 1) % 100 == 0:
        print(f"  Fetched {i + 1}/{len(game_pks)} dates...")
    game_dates[pk] = get_game_date(pk)

# Find min and max dates
valid_dates = {pk: date for pk, date in game_dates.items() if date is not None}

if not valid_dates:
    print("\n❌ Could not fetch any game dates")
    exit(1)

min_date = min(valid_dates.values())
max_date = max(valid_dates.values())

# Find which game_pks correspond to these dates
min_pk = [pk for pk, date in valid_dates.items() if date == min_date][0]
max_pk = [pk for pk, date in valid_dates.items() if date == max_date][0]

print("\n" + "=" * 60)
print("DATE RANGE")
print("=" * 60)
print(f"Earliest game (pk={min_pk}): {min_date}")
print(f"Latest game (pk={max_pk}):   {max_date}")
print(f"Total unique dates: {len(set(valid_dates.values()))}")
print("=" * 60)
