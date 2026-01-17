import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import tensorflow as tf
from src.models.pitch_sequence_baseline import build_pitch_model

# Paths
TRAIN_DIR = Path("data/gold/train")
VAL_DIR = Path("data/gold/val")
PITCHER_LOOKUP = Path("data/silver/dims/pitcher_lookup.parquet")
BATTER_LOOKUP = Path("data/silver/dims/batter_lookup.parquet")
PITCH_LOOKUP = Path("data/silver/dims/pitch_lookup.parquet")
MODEL_SAVE_PATH = Path("models/baseline_pitch_type")

# Training hyperparameters
BATCH_SIZE = 512
EPOCHS = 50
PREDICT_TARGET = "pitch_type"  # or "pitch_bucket"

def load_vocab_sizes():
    """Load lookup tables to determine vocabulary sizes."""
    pitcher_lookup = pd.read_parquet(PITCHER_LOOKUP)
    batter_lookup = pd.read_parquet(BATTER_LOOKUP)
    pitch_lookup = pd.read_parquet(PITCH_LOOKUP)

    # +1 for index 0 (used for padding/unknown in lagged features)
    n_pitchers = len(pitcher_lookup) + 1
    n_batters = len(batter_lookup) + 1

    # For output classes, use actual count (not max+1)
    # Since we shift targets to be 0-indexed in the dataset
    n_pitch_types = len(pitch_lookup)  # 13 pitch types
    n_pitch_buckets = pitch_lookup["pitch_bucket_idx"].nunique()  # 3 buckets

    print(f"Vocabulary sizes:")
    print(f"  Pitchers (embeddings): {n_pitchers}")
    print(f"  Batters (embeddings): {n_batters}")
    print(f"  Pitch types (output classes): {n_pitch_types}")
    print(f"  Pitch buckets (output classes): {n_pitch_buckets}")

    return n_pitchers, n_batters, n_pitch_types, n_pitch_buckets


def load_parquet_files(directory):
    """Load all parquet files from a directory and concatenate them."""
    parquet_files = sorted(directory.glob("*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {directory}")

    print(f"Loading {len(parquet_files)} files from {directory}")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    print(f"  Total samples: {len(df)}")
    return df


def df_to_tf_dataset(df, target_col, batch_size, shuffle=True):
    """Convert a pandas DataFrame to a TensorFlow dataset."""

    # Define the feature columns expected by the model
    feature_cols = {
        "inning": df["inning"].values.astype("float32"),
        "balls_before_pitch": df["balls_before_pitch"].values.astype("float32"),
        "strikes_before_pitch": df["strikes_before_pitch"].values.astype("float32"),
        "pitcher_is_home": df["pitcher_is_home"].values.astype("float32"),
        "pitcher_is_right": df["pitcher_is_right"].values.astype("float32"),
        "batter_is_right": df["batter_is_right"].values.astype("float32"),
        "pitch_num_in_pa": df["pitch_num_in_pa"].values.astype("float32"),
        "pitcher_idx": df["pitcher_idx"].values.astype("int32"),
        "batter_idx": df["batter_idx"].values.astype("int32"),
        "pitch_1_idx": df["pitch_1_idx"].values.astype("int32"),
        "pitch_2_idx": df["pitch_2_idx"].values.astype("int32"),
        "pitch_3_idx": df["pitch_3_idx"].values.astype("int32"),
        "pitch_1_bucket_idx": df["pitch_1_bucket_idx"].values.astype("int32"),
        "pitch_2_bucket_idx": df["pitch_2_bucket_idx"].values.astype("int32"),
        "pitch_3_bucket_idx": df["pitch_3_bucket_idx"].values.astype("int32"),
    }

    # Target variable - shift to 0-indexed for sparse_categorical_crossentropy
    # Original indices start at 1, TensorFlow expects 0-indexed
    target = (df[target_col].values - 1).astype("int32")

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((feature_cols, target))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000, seed=42)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def main():
    print("=" * 60)
    print("Training Baseline Pitch Sequence Model")
    print("=" * 60)

    # Load vocabulary sizes
    n_pitchers, n_batters, n_pitch_types, n_pitch_buckets = load_vocab_sizes()

    # Load data
    print("\nLoading training data...")
    train_df = load_parquet_files(TRAIN_DIR)

    print("\nLoading validation data...")
    val_df = load_parquet_files(VAL_DIR)

    # Determine target column based on prediction type
    if PREDICT_TARGET == "pitch_type":
        target_col = "pitch_idx"
    elif PREDICT_TARGET == "pitch_bucket":
        target_col = "pitch_bucket_idx"
    else:
        raise ValueError(f"Invalid PREDICT_TARGET: {PREDICT_TARGET}")

    print(f"\nPrediction target: {PREDICT_TARGET} (column: {target_col})")

    # Create TensorFlow datasets
    print("\nCreating TensorFlow datasets...")
    train_ds = df_to_tf_dataset(train_df, target_col, BATCH_SIZE, shuffle=True)
    val_ds = df_to_tf_dataset(val_df, target_col, BATCH_SIZE, shuffle=False)

    # Build model
    print("\nBuilding model...")
    model = build_pitch_model(
        n_pitchers=n_pitchers,
        n_batters=n_batters,
        n_pitch_types=n_pitch_types,
        n_pitch_buckets=n_pitch_buckets,
        predict=PREDICT_TARGET,
        pitcher_emb_dim=32,
        batter_emb_dim=32,
        pitch_emb_dim=16,
        bucket_emb_dim=8,
        hidden_units=(256, 128),
        dropout=0.2,
    )

    print("\nModel summary:")
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH / "checkpoint.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
    ]

    # Create model directory
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # Train model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    print("\nSaving final model...")
    model.save(str(MODEL_SAVE_PATH / "final_model.keras"))

    # Print final metrics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final training accuracy: {history.history['sparse_categorical_accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_sparse_categorical_accuracy'][-1]:.4f}")
    print(f"Best validation accuracy: {max(history.history['val_sparse_categorical_accuracy']):.4f}")
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
