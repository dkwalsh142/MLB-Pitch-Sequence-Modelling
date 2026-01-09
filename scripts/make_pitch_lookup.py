import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from src.parquet_functions import write_parquet_file
from src.dims import update_lookup_table

PITCH_TYPES = {"FF", "FT", "SI", "FC", "SL", "CU", "KC", "SV", "ST", "CH", "FS", "FO"}

FASTBALLS = {"FF", "FT", "SI", "FC"}
BREAKING  = {"SL", "CU", "KC", "SV", "ST"}
OFFSPEED  = {"CH", "FS", "FO"}

PITCH_INDEXES = {"FF": 1, "FT": 2, "SI": 3, "FC": 4, "SL": 5, "CU": 6, "KC": 7, "SV": 8, "ST": 9, "CH": 10, "FS": 11, "FO": 12}
BUCKET_INDEXES = {"fastball": 1, "breaking": 2, "offspeed": 3}

pitch_data = []

for pitch in PITCH_TYPES:

    pitch_code = pitch

    pitch_idx = PITCH_INDEXES[pitch]

    if pitch in FASTBALLS:
        pitch_bucket = "fastball"
        pitch_bucket_idx = 1
    elif pitch in BREAKING:
        pitch_bucket = "breaking"
        pitch_bucket_idx = 2
    elif pitch in OFFSPEED:
        pitch_bucket = "offspeed"
        pitch_bucket_idx = 3

    pitch_data.append({
        "pitch_code": pitch_code,
        "pitch_idx": pitch_idx,
        "pitch_bucket": pitch_bucket,
        "pitch_bucket_idx": pitch_bucket_idx
    })

df = pd.DataFrame(pitch_data)

write_parquet_file(
        df, 
        f"data/silver/dims/pitch_lookup.parquet"
        )