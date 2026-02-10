import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pandas as pd
import tensorflow as tf
import numpy as np
from src.models.pitch_sequence_ten_lags import build_pitch_model
from contextlib import redirect_stdout
import io

# Paths
VAL_DIR = Path("data/gold/val")
PITCHER_LOOKUP = Path("data/silver/dims/pitcher_lookup.parquet")
BATTER_LOOKUP = Path("data/silver/dims/batter_lookup.parquet")
PITCH_LOOKUP = Path("data/silver/dims/pitch_lookup.parquet")
MODEL_PATH = Path("models/ten_lags_pitch_type/final_model.keras")
RESULTS_OUTPUT = Path("models/ten_lags_pitch_type/validation_results.txt")

# Hyperparameters
BATCH_SIZE = 1024
PREDICT_TARGET = "pitch_type"  # or "pitch_bucket"


class TeeOutput:
    """Write to both console and file simultaneously."""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def load_vocab_sizes():
    """Load lookup tables to determine vocabulary sizes."""
    pitcher_lookup = pd.read_parquet(PITCHER_LOOKUP)
    batter_lookup = pd.read_parquet(BATTER_LOOKUP)
    pitch_lookup = pd.read_parquet(PITCH_LOOKUP)

    # +1 for index 0 (used for padding/unknown in lagged features)
    n_pitchers = len(pitcher_lookup) + 1
    n_batters = len(batter_lookup) + 1

    # Pitch types: indices 1-13, but we need vocab size 14 to include index 0 (padding)
    n_pitch_types = len(pitch_lookup)  # 13 pitch types (for output layer)
    n_pitch_types_vocab = len(pitch_lookup) + 1  # 14 (for embeddings, includes 0)

    n_pitch_buckets = pitch_lookup["pitch_bucket_idx"].nunique()  # 3 buckets
    n_pitch_buckets_vocab = pitch_lookup["pitch_bucket_idx"].nunique() + 1  # 4

    return n_pitchers, n_batters, n_pitch_types, n_pitch_buckets, n_pitch_types_vocab, n_pitch_buckets_vocab, pitch_lookup


def load_parquet_files(directory):
    """Load all parquet files from a directory and concatenate them."""
    parquet_files = sorted(directory.glob("*.parquet"))

    if not parquet_files:
        raise ValueError(f"No parquet files found in {directory}")

    print(f"Loading {len(parquet_files)} files from {directory}")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)

    print(f"  Total samples: {len(df):,}")
    return df


def df_to_tf_dataset(df, target_col, batch_size):
    """Convert a pandas DataFrame to a TensorFlow dataset."""

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
        "pitch_4_idx": df["pitch_4_idx"].values.astype("int32"),
        "pitch_5_idx": df["pitch_5_idx"].values.astype("int32"),
        "pitch_6_idx": df["pitch_6_idx"].values.astype("int32"),
        "pitch_7_idx": df["pitch_7_idx"].values.astype("int32"),
        "pitch_8_idx": df["pitch_8_idx"].values.astype("int32"),
        "pitch_9_idx": df["pitch_9_idx"].values.astype("int32"),
        "pitch_10_idx": df["pitch_10_idx"].values.astype("int32"),
        "pitch_1_bucket_idx": df["pitch_1_bucket_idx"].values.astype("int32"),
        "pitch_2_bucket_idx": df["pitch_2_bucket_idx"].values.astype("int32"),
        "pitch_3_bucket_idx": df["pitch_3_bucket_idx"].values.astype("int32"),
    }

    # Target variable - keep 1-indexed (1-13 for pitch types, 1-3 for buckets)
    target = df[target_col].values.astype("int32")

    dataset = tf.data.Dataset.from_tensor_slices((feature_cols, target))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def calculate_top_k_accuracy(y_true, y_pred_probs, k=3):
    """Calculate top-k accuracy."""
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return correct.mean()


def analyze_per_class_performance(y_true, y_pred, pitch_lookup):
    """Analyze performance for each pitch type."""
    from sklearn.metrics import classification_report, confusion_matrix

    # Get all pitch type codes (e.g., 'FF', 'SL', 'CH') - sorted by pitch_idx (1-13)
    pitch_lookup_sorted = pitch_lookup.sort_values('pitch_idx')

    # Get unique classes that actually appear in predictions (1-indexed: 1-13)
    unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))

    # Map pitch indices to names
    pitch_names = [pitch_lookup_sorted[pitch_lookup_sorted['pitch_idx'] == idx]['pitch_code'].values[0]
                   for idx in unique_classes]

    print("\n" + "=" * 60)
    print("PER-CLASS PERFORMANCE")
    print("=" * 60)

    # Show distribution of actual pitch types in validation set
    print("\nPitch Type Distribution in Validation Set:")
    print("-" * 50)
    unique, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)

    # Sort by frequency (most common first)
    sorted_indices = np.argsort(-counts)

    print(f"{'Pitch Type':<15} {'Count':>10} {'Percentage':>12}")
    print("-" * 50)
    for idx in sorted_indices:
        pitch_idx = unique[idx]
        count = counts[idx]
        percentage = (count / total) * 100
        # pitch_idx is 1-indexed (1-13), lookup directly from pitch_lookup
        pitch_code = pitch_lookup_sorted[pitch_lookup_sorted['pitch_idx'] == pitch_idx]['pitch_code'].values[0]
        print(f"{pitch_code:<15} {count:>10,} {percentage:>11.2f}%")
    print("-" * 50)
    print(f"{'TOTAL':<15} {total:>10,} {100.0:>11.2f}%")
    print()

    # Classification report - use labels parameter to specify which classes to include
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=unique_classes, target_names=pitch_names, zero_division=0))

    # Confusion matrix - use labels parameter to match classification report
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

    # Print confusion matrix with labels
    row_label_width = 10  # "{name:>8} | " = 8 + 2 + 1 space
    col_width = 9  # Width for each column (includes the asterisk/space)
    print("\n" + " " * row_label_width + "Predicted")
    print(" " * row_label_width, end="")
    for name in pitch_names:
        print(f"{name[:8]:>{col_width}}", end="")
    print()

    for i, name in enumerate(pitch_names):
        print(f"{name[:8]:>8} |", end="")
        for j in range(len(pitch_names)):
            # Add asterisk to diagonal instead of ANSI bold codes
            if i == j:
                print(f"{cm[i, j]:>8}*", end="")
            else:
                print(f"{cm[i, j]:>9}", end="")
        print(f" | {cm[i].sum():>6}")

    return cm


def analyze_predictions_by_context(df, y_true, y_pred, y_pred_probs):
    """Analyze predictions based on game context."""
    df = df.copy()
    df['correct'] = (y_true == y_pred)
    df['confidence'] = np.max(y_pred_probs, axis=1)

    print("\n" + "=" * 60)
    print("PERFORMANCE BY CONTEXT")
    print("=" * 60)

    # By count
    print("\nAccuracy by Count:")
    for balls in sorted(df['balls_before_pitch'].unique()):
        for strikes in sorted(df['strikes_before_pitch'].unique()):
            mask = (df['balls_before_pitch'] == balls) & (df['strikes_before_pitch'] == strikes)
            if mask.sum() > 0:
                acc = df[mask]['correct'].mean()
                count = mask.sum()
                print(f"  {int(balls)}-{int(strikes)}: {acc:.4f} ({count:,} samples)")

    # By inning
    print("\nAccuracy by Inning:")
    for inning in sorted(df['inning'].unique()):
        mask = df['inning'] == inning
        if mask.sum() > 0:
            acc = df[mask]['correct'].mean()
            count = mask.sum()
            print(f"  Inning {int(inning):2d}: {acc:.4f} ({count:,} samples)")

    # By pitch number in plate appearance
    print("\nAccuracy by Pitch Number in PA:")
    for pitch_num in sorted(df['pitch_num_in_pa'].unique()):
        mask = df['pitch_num_in_pa'] == pitch_num
        if mask.sum() > 0:
            acc = df[mask]['correct'].mean()
            count = mask.sum()
            if pitch_num <= 10:  # Only show first 10 pitches
                print(f"  Pitch {int(pitch_num):2d}: {acc:.4f} ({count:,} samples)")

    # Confidence analysis
    print("\nPrediction Confidence Analysis:")
    print(f"  Mean confidence: {df['confidence'].mean():.4f}")
    print(f"  Median confidence: {df['confidence'].median():.4f}")

    # Accuracy by confidence quartile
    print("\nAccuracy by Confidence Quartile:")
    df['conf_quartile'] = pd.qcut(df['confidence'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    for quartile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']:
        mask = df['conf_quartile'] == quartile
        acc = df[mask]['correct'].mean()
        count = mask.sum()
        avg_conf = df[mask]['confidence'].mean()
        print(f"  {quartile}: {acc:.4f} (avg conf: {avg_conf:.4f}, {count:,} samples)")


def main():
    # Set up output to write to both console and file
    RESULTS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tee = TeeOutput(RESULTS_OUTPUT)
    original_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 60)
        print("Validating Ten Lags Pitch Sequence Model")
        print("=" * 60)

        # Check if model exists
        if not MODEL_PATH.exists():
            print(f"\n❌ Error: Model not found at {MODEL_PATH}")
            print("Please train the model first using train_ten_lags.py")
            return

        # Check if validation data exists
        if not VAL_DIR.exists() or not list(VAL_DIR.glob("*.parquet")):
            print(f"\n❌ Error: Validation data not found at {VAL_DIR}")
            print("Please create validation data using make_gold.py")
            return

        # Load vocabulary sizes and lookup table
        print("\nLoading vocabulary and lookup tables...")
        n_pitchers, n_batters, n_pitch_types, n_pitch_buckets, n_pitch_types_vocab, n_pitch_buckets_vocab, pitch_lookup = load_vocab_sizes()

        # Load validation data
        print("\nLoading validation data...")
        val_df = load_parquet_files(VAL_DIR)

        # Determine target column
        if PREDICT_TARGET == "pitch_type":
            target_col = "pitch_idx"
        elif PREDICT_TARGET == "pitch_bucket":
            target_col = "pitch_bucket_idx"
        else:
            raise ValueError(f"Invalid PREDICT_TARGET: {PREDICT_TARGET}")

        print(f"\nPrediction target: {PREDICT_TARGET} (column: {target_col})")

        # Create TensorFlow dataset
        print("\nCreating TensorFlow dataset...")
        val_ds = df_to_tf_dataset(val_df, target_col, BATCH_SIZE)

        # Load model
        print("\nLoading trained model...", MODEL_PATH)
        model = tf.keras.models.load_model(MODEL_PATH)

        print("\nModel summary:")
        model.summary()

        # Evaluate model
        print("\n" + "=" * 60)
        print("EVALUATING MODEL ON VALIDATION SET")
        print("=" * 60)

        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        print(f"Evaluation complete.")

        print("\n" + "=" * 60)
        print("BASIC METRICS")
        print("=" * 60)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Get predictions
        print("\nGenerating predictions...")
        y_pred_logits = model.predict(val_ds, verbose=0)
        print("Predictions complete.")
        # Convert logits to probabilities using softmax
        y_pred_probs = tf.nn.softmax(y_pred_logits).numpy()
        y_pred = np.argmax(y_pred_probs, axis=1)
        # Keep targets 1-indexed (1-13 for pitch types, 1-3 for buckets)
        y_true = val_df[target_col].values.astype("int32")

        # Calculate top-k accuracies
        print("\n" + "=" * 60)
        print("TOP-K ACCURACY")
        print("=" * 60)
        top_1_acc = calculate_top_k_accuracy(y_true, y_pred_probs, k=1)
        top_3_acc = calculate_top_k_accuracy(y_true, y_pred_probs, k=3)
        top_5_acc = calculate_top_k_accuracy(y_true, y_pred_probs, k=5)

        print(f"Top-1 Accuracy: {top_1_acc:.4f}")
        print(f"Top-3 Accuracy: {top_3_acc:.4f}")
        print(f"Top-5 Accuracy: {top_5_acc:.4f}")

        # Analyze per-class performance
        if PREDICT_TARGET == "pitch_type":
            cm = analyze_per_class_performance(y_true, y_pred, pitch_lookup)

        # Analyze by context
        analyze_predictions_by_context(val_df, y_true, y_pred, y_pred_probs)

        # Random sample predictions
        print("\n" + "=" * 60)
        print("SAMPLE PREDICTIONS")
        print("=" * 60)

        # Show 10 random predictions
        sample_indices = np.random.choice(len(val_df), size=min(10, len(val_df)), replace=False)

        for idx in sample_indices:
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            confidence = y_pred_probs[idx, pred_label]

            # Get top 3 predictions
            top_3_indices = np.argsort(y_pred_probs[idx])[-3:][::-1]

            if PREDICT_TARGET == "pitch_type":
                # Note: y_true and y_pred are 1-indexed (1-13), matching pitch_idx in lookup table
                true_code = pitch_lookup[pitch_lookup['pitch_idx'] == true_label]['pitch_code'].values[0]
                pred_code = pitch_lookup[pitch_lookup['pitch_idx'] == pred_label]['pitch_code'].values[0]

                print(f"\nSample {idx}:")
                print(f"  Context: {int(val_df.iloc[idx]['balls_before_pitch'])}-{int(val_df.iloc[idx]['strikes_before_pitch'])} count, "
                      f"Inning {int(val_df.iloc[idx]['inning'])}, Pitch #{int(val_df.iloc[idx]['pitch_num_in_pa'])}")
                print(f"  True: {true_code}")
                print(f"  Predicted: {pred_code} ({confidence:.3f})")
                print(f"  Top 3 predictions:")
                for i, pred_idx in enumerate(top_3_indices):
                    pred_code_i = pitch_lookup[pitch_lookup['pitch_idx'] == pred_idx]['pitch_code'].values[0]
                    print(f"    {i+1}. {pred_code_i}: {y_pred_probs[idx, pred_idx]:.3f}")

        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE!")
        print("=" * 60)
        print(f"\nResults saved to: {RESULTS_OUTPUT}")

    finally:
        # Restore stdout and close file
        sys.stdout = original_stdout
        tee.close()
        print(f"\n✅ Validation results saved to: {RESULTS_OUTPUT}")


if __name__ == "__main__":
    main()
