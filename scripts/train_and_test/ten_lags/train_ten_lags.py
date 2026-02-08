import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from src.models.pitch_sequence_ten_lags import build_pitch_model

# Paths
TRAIN_DIR = Path("data/gold/train")
VAL_DIR = Path("data/gold/val")
PITCHER_LOOKUP = Path("data/silver/dims/pitcher_lookup.parquet")
BATTER_LOOKUP = Path("data/silver/dims/batter_lookup.parquet")
PITCH_LOOKUP = Path("data/silver/dims/pitch_lookup.parquet")
MODEL_SAVE_PATH = Path("models/ten_lags_pitch_type")

# Training hyperparameters
BATCH_SIZE = 1024
EPOCHS = 100
PREDICT_TARGET = "pitch_type"  # or "pitch_bucket"

def load_vocab_sizes():
    """Load lookup tables to determine vocabulary sizes."""
    pitcher_lookup = pd.read_parquet(PITCHER_LOOKUP)
    batter_lookup = pd.read_parquet(BATTER_LOOKUP)
    pitch_lookup = pd.read_parquet(PITCH_LOOKUP)

    # +1 for index 0 (used for padding/unknown in lagged features)
    n_pitchers = len(pitcher_lookup) + 1
    n_batters = len(batter_lookup) + 1

    # Pitch types: indices 1-13, but we need vocab size 14 to include index 0 (padding)
    # Output classes: still 1-13 (we don't predict padding)
    n_pitch_types = len(pitch_lookup)  # 13 pitch types (for output layer)
    n_pitch_types_vocab = len(pitch_lookup) + 1  # 14 (for embeddings, includes 0)

    n_pitch_buckets = pitch_lookup["pitch_bucket_idx"].nunique()  # 3 buckets (for output)
    n_pitch_buckets_vocab = pitch_lookup["pitch_bucket_idx"].nunique() + 1  # 4 (for embeddings, includes 0)

    print(f"Vocabulary sizes:")
    print(f"  Pitchers (embeddings): {n_pitchers}")
    print(f"  Batters (embeddings): {n_batters}")
    print(f"  Pitch types (output classes): {n_pitch_types} (vocab: {n_pitch_types_vocab})")
    print(f"  Pitch buckets (output classes): {n_pitch_buckets} (vocab: {n_pitch_buckets_vocab})")

    return n_pitchers, n_batters, n_pitch_types, n_pitch_buckets, n_pitch_types_vocab, n_pitch_buckets_vocab


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
    # We'll use sparse_categorical_crossentropy with num_classes parameter
    target = df[target_col].values.astype("int32")

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((feature_cols, target))

    # CRITICAL OPTIMIZATION: Cache the dataset to avoid recreating it each epoch
    # This keeps the data in memory after the first epoch
    dataset = dataset.cache()

    if shuffle:
        # Use large shuffle buffer (50k is a good balance between randomization and memory)
        # Full dataset shuffle (len(df)) can cause memory issues
        shuffle_buffer = min(50000, len(df))
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=42, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def main():
    print("=" * 60)
    print("Training Ten Lags Pitch Sequence Model")
    print("=" * 60)

    # Load vocabulary sizes
    n_pitchers, n_batters, n_pitch_types, n_pitch_buckets, n_pitch_types_vocab, n_pitch_buckets_vocab = load_vocab_sizes()

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
        n_pitch_types_vocab=n_pitch_types_vocab,
        n_pitch_buckets_vocab=n_pitch_buckets_vocab,
        predict=PREDICT_TARGET,
        pitcher_emb_dim=64,
        batter_emb_dim=64,
        pitch_emb_dim=24,
        bucket_emb_dim=12,
        hidden_units=(512, 256, 128),
        dropout=0.3,
    )

    print("\nModel summary:")
    model.summary()

    # Custom callback to show time per epoch and live plot
    class TrainingPlotCallback(tf.keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            plt.ion()  # Enable interactive mode
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 5))
            self.fig.suptitle('Training Progress')
            self.train_losses = []
            self.val_losses = []
            self.train_accs = []
            self.val_accs = []

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start_time
            train_loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            train_acc = logs.get('sparse_categorical_accuracy', 0)
            val_acc = logs.get('val_sparse_categorical_accuracy', 0)
            print(f"  -> Epoch {epoch + 1} took {epoch_time:.1f}s | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            epochs = range(1, len(self.train_losses) + 1)

            # Update loss plot
            self.ax1.clear()
            self.ax1.plot(epochs, self.train_losses, 'b-o', markersize=3, label='Training Loss')
            self.ax1.plot(epochs, self.val_losses, 'r-o', markersize=3, label='Validation Loss')
            self.ax1.set_xlabel('Epoch')
            self.ax1.set_ylabel('Loss')
            self.ax1.set_title('Training vs Validation Loss')
            self.ax1.legend()
            self.ax1.grid(True)

            # Update accuracy plot
            self.ax2.clear()
            self.ax2.plot(epochs, self.train_accs, 'b-o', markersize=3, label='Training Accuracy')
            self.ax2.plot(epochs, self.val_accs, 'r-o', markersize=3, label='Validation Accuracy')
            self.ax2.set_xlabel('Epoch')
            self.ax2.set_ylabel('Accuracy')
            self.ax2.set_title('Training vs Validation Accuracy')
            self.ax2.legend()
            self.ax2.grid(True)

            self.fig.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        def on_train_end(self, logs=None):
            # Save final plot
            plot_path = MODEL_SAVE_PATH / "training_curves.png"
            self.fig.savefig(plot_path, dpi=150)
            print(f"\nTraining curves saved to: {plot_path}")
            plt.ioff()  # Disable interactive mode

    # Custom callback to save best model per 10-epoch window
    class BestPerWindowCallback(tf.keras.callbacks.Callback):
        def __init__(self, save_dir, window_size=10):
            super().__init__()
            self.save_dir = Path(save_dir)
            self.window_size = window_size
            self.best_val_loss_in_window = float('inf')
            self.window_start = 1

        def on_epoch_end(self, epoch, logs=None):
            epoch_num = epoch + 1  # 1-indexed
            val_loss = logs.get('val_loss', float('inf'))

            if val_loss < self.best_val_loss_in_window:
                self.best_val_loss_in_window = val_loss
                save_path = self.save_dir / f"best_epoch_{self.window_start}-{self.window_start + self.window_size - 1}.keras"
                self.model.save(str(save_path))
                print(f"  -> Saved best model for epochs {self.window_start}-{self.window_start + self.window_size - 1} (val_loss: {val_loss:.4f})")

            # Reset window
            if epoch_num % self.window_size == 0:
                self.best_val_loss_in_window = float('inf')
                self.window_start = epoch_num + 1

    # Callbacks
    callbacks = [
        TrainingPlotCallback(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
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
        # Save best model overall
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH / "best_overall.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        # Save best model per 10-epoch window
        BestPerWindowCallback(MODEL_SAVE_PATH, window_size=10),
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

    # Save training history to CSV
    history_df = pd.DataFrame({
        "epoch": range(1, len(history.history["loss"]) + 1),
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "train_accuracy": history.history["sparse_categorical_accuracy"],
        "val_accuracy": history.history["val_sparse_categorical_accuracy"],
    })
    history_path = MODEL_SAVE_PATH / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"\nTraining history saved to: {history_path}")

    # Print final metrics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final training accuracy: {history.history['sparse_categorical_accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_sparse_categorical_accuracy'][-1]:.4f}")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"Best validation accuracy: {max(history.history['val_sparse_categorical_accuracy']):.4f}")
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")



if __name__ == "__main__":
    main()
