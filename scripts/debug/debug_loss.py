import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import tensorflow as tf
from src.models.pitch_sequence_baseline import build_pitch_model

# Load a small sample
train_file = next(Path("data/gold/train").glob("*.parquet"))
df = pd.read_parquet(train_file).head(100)

print(f"Sample size: {len(df)}")
print(f"pitch_idx range: [{df['pitch_idx'].min()}, {df['pitch_idx'].max()}]")

# Create a tiny dataset
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

# Keep targets 1-indexed (no shift needed)
target = df["pitch_idx"].values.astype("int32")
print(f"Target range (1-indexed): [{target.min()}, {target.max()}]")

# Build model
model = build_pitch_model(
    n_pitchers=1000,
    n_batters=1000,
    n_pitch_types=13,  # Output classes
    n_pitch_buckets=3,  # Output classes
    n_pitch_types_vocab=14,  # Embedding vocab (includes 0)
    n_pitch_buckets_vocab=4,  # Embedding vocab (includes 0)
    predict="pitch_type",
    pitcher_emb_dim=64,
    batter_emb_dim=64,
    pitch_emb_dim=24,
    bucket_emb_dim=12,
    hidden_units=(512, 256, 128),
    dropout=0.3,
)

# Evaluate loss on this batch
loss = model.evaluate(feature_cols, target, batch_size=100, verbose=0)
print(f"\nModel loss on 100 samples: {loss[0]:.4f}")
print(f"Expected loss (random): ~2.56")
print(f"Ratio: {loss[0] / 2.56:.1f}x")

# Check model compilation
print(f"\nModel loss function: {model.loss}")
print(f"Model compiled with reduction: {model.compiled_loss._loss_metric.reduction}")
