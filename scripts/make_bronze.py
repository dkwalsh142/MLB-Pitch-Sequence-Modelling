import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.mlb_api import get_game_pks
from src.mlb_api import get_pitch_feed
from src.parquet_functions import write_parquet_file

start_date = "2024-04-01"
end_date = "2024-04-07"

game_pks = get_game_pks(start_date, end_date)
print(len(game_pks))
print(game_pks)

for game_pk in game_pks:
    df = get_pitch_feed(game_pk)
    write_parquet_file(
        df, 
        f"data/bronze/game_{game_pk}.parquet"
        )
    

# df_read = pd.read_parquet(
#     "data/bronze/game_746817.parquet",
#     engine="pyarrow"
# )
# print(df_read)




