import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.mlb_api import get_game_pks
from src.mlb_api import get_pitch_feed
from src.mlb_api import print_game_dates
from src.parquet_functions import write_parquet_file

start_date = "2024-04-01"
end_date = "2024-04-07"

game_pks = get_game_pks(start_date, end_date)
print(len(game_pks))
#print(game_pks)
#print_game_dates(game_pks)



for game_pk in game_pks:
    df = get_pitch_feed(game_pk)
    write_parquet_file(
        df, 
        f"data/bronze/game_{game_pk}.parquet"
        )


#      Column               Dtype 
# ---  ------               ----- 
#  0   game_pk              int64 
#  1   ab_idx               int64 
#  2   pitch_num_in_pa      int64 
#  3   inning               int64 
#  4   pitcher_id           int64 
#  5   batter_id            int64 
#  6   pitcher_hand         object
#  7   batter_hand          object
#  8   pitcher_is_home      bool  
#  9   balls_after_pitch    int64 
#  10  strikes_after_pitch  int64 
#  11  pitch_type           object




