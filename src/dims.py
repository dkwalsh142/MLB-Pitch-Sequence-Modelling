from pathlib import Path
import pandas as pd

def update_lookup_table(
    lookup_path: Path,
    id_col: str,
    idx_col: str,
    incoming_ids: pd.Series,
) -> pd.DataFrame:
    """
    Append-only lookup table stored as Parquet.

    - Loads existing lookup (if present)
    - Adds new IDs not already present
    - Assigns new indices at the end
    - Writes back to Parquet
    - Returns updated lookup DataFrame
    """
    lookup_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize incoming IDs
    incoming_unique = (
        pd.Series(incoming_ids)
        .dropna()
        .astype("int64")
        .unique()
    )

    # Load existing or initialize
    if lookup_path.exists():
        lookup = pd.read_parquet(lookup_path)
        # Ensure expected columns exist
        if not {id_col, idx_col}.issubset(set(lookup.columns)):
            raise ValueError(f"Lookup at {lookup_path} missing required cols {id_col}, {idx_col}")
        existing_ids = set(lookup[id_col].astype("int64").tolist())
        start_idx = int(lookup[idx_col].max()) + 1 if len(lookup) else 0
    else:
        lookup = pd.DataFrame(columns=[id_col, idx_col])
        existing_ids = set()
        start_idx = 0

    # Compute what’s new
    new_ids = sorted(set(incoming_unique) - existing_ids)

    # Append if needed
    if new_ids:
        additions = pd.DataFrame({
            id_col: new_ids,
            idx_col: range(start_idx, start_idx + len(new_ids)),
        })

        lookup = pd.concat([lookup, additions], ignore_index=True)

        # Enforce stable dtypes
        lookup[id_col] = lookup[id_col].astype("int32")
        lookup[idx_col] = lookup[idx_col].astype("int32")

        # Write back to Parquet
        lookup.to_parquet(lookup_path, index=False)

    else:
        # Still enforce dtypes if non-empty
        if len(lookup):
            lookup[id_col] = lookup[id_col].astype("int32")
            lookup[idx_col] = lookup[idx_col].astype("int32")

    # Safety check: uniqueness
    if len(lookup) and not lookup[id_col].is_unique:
        raise ValueError(f"{id_col} is not unique in {lookup_path}")

    return lookup
