import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def write_parquet_file(df: pd.DataFrame, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, preserve_index=False)

    pq.write_table(
        table,
        path,
        compression="snappy"
    )