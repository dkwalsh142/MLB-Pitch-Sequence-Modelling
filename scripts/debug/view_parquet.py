import argparse
import pandas as pd

#debug tool to view parquets

PARQUET_PATH = "data/bronze/game_716352.parquet"

def inspect_df(
    df: pd.DataFrame,
    *,
    n_head: int = 10,
    n_tail: int = 5,
    print_columns: bool = True,
    print_info: bool = True,
    print_dtypes: bool = False,
    show_head: bool = True,
    show_tail: bool = True,
    transpose_head: bool = False,
    max_columns: int | None = None,   # None => show all columns
    max_rows: int | None = None,      # None => pandas default (prevents huge dumps)
    max_colwidth: int | None = None,  # None => no truncation of long strings
) -> None:
    """
    Safe, repeatable DataFrame inspection for wide / large tables.
    Set max_columns=None to display all columns in head/tail outputs.
    """

    # Save current options so we can restore them (avoid global side-effects).
    old_max_cols = pd.get_option("display.max_columns")
    old_max_rows = pd.get_option("display.max_rows")
    old_max_colwidth = pd.get_option("display.max_colwidth")
    old_width = pd.get_option("display.width")

    try:
        # Apply requested display options.
        if max_columns is None:
            pd.set_option("display.width", None)  # Disable width limit only when showing all columns
            pd.set_option("display.max_columns", None)  # Show all columns
        elif max_columns is not None:
            pd.set_option("display.max_columns", max_columns)
        if max_rows is not None:
            pd.set_option("display.max_rows", max_rows)
        if max_colwidth is not None:
            pd.set_option("display.max_colwidth", max_colwidth)

        if print_info:
            print("\n=== df.info() ===")
            df.info()

        if print_columns:
            print("\n=== Columns (in order) ===")
            for i, c in enumerate(df.columns, start=1):
                print(f"{i:>3}. {c}")

        if print_dtypes:
            print("\n=== dtypes ===")
            print(df.dtypes)

        if show_head:
            print(f"\n=== Head ({n_head}) ===")
            head_df = df.head(n_head)
            print(head_df.T if transpose_head else head_df)

        if show_tail:
            print(f"\n=== Tail ({n_tail}) ===")
            print(df.tail(n_tail))

    finally:
        # Restore pandas options.
        pd.set_option("display.max_columns", old_max_cols)
        pd.set_option("display.max_rows", old_max_rows)
        pd.set_option("display.max_colwidth", old_max_colwidth)
        pd.set_option("display.width", old_width)


def main() -> None:
    parser = argparse.ArgumentParser(description="View parquet files with optional column display control")
    parser.add_argument(
        "--all-columns",
        action="store_true",
        help="Show all columns (default: uses pandas default column limit)"
    )
    args = parser.parse_args()

    df_read = pd.read_parquet(PARQUET_PATH, engine="pyarrow")

    inspect_df(
        df_read,
        n_head=10,
        n_tail=10,
        print_columns=True,
        print_info=True,
        print_dtypes=False,
        show_head=True,
        show_tail=True,
        transpose_head=False,
        max_columns=None if args.all_columns else 20,  # Default to 20 columns max
        max_colwidth=None,    # do not truncate long strings
        # max_rows=None       # keep default to avoid massive prints
    )


if __name__ == "__main__":
    main()
