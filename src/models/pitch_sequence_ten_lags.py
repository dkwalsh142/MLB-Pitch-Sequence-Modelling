import tensorflow as tf
from tensorflow.keras import layers as L

def build_pitch_model(
    n_pitchers: int,
    n_batters: int,
    n_pitch_types: int,           # Number of output classes (e.g., 13)
    n_pitch_buckets: int,         # Number of bucket classes (e.g., 3)
    n_pitch_types_vocab: int,     # Vocab size for embeddings (e.g., 14, includes 0 for padding)
    n_pitch_buckets_vocab: int,   # Vocab size for bucket embeddings (e.g., 4, includes 0)
    predict: str = "pitch_type",  # "pitch_type" or "pitch_bucket"
    pitcher_emb_dim: int = 32,
    batter_emb_dim: int = 32,
    pitch_emb_dim: int = 16,
    bucket_emb_dim: int = 8,
    hidden_units=(256, 128),
    dropout=0.2,
):
    """
    Baseline model:
      - Embeddings for IDs and lagged pitch classes
      - Dense tower for prediction
    """

    # -------------------------
    # Numeric / boolean inputs
    # -------------------------
    inning = L.Input(shape=(1,), dtype=tf.float32, name="inning")
    balls = L.Input(shape=(1,), dtype=tf.float32, name="balls_before_pitch")
    strikes = L.Input(shape=(1,), dtype=tf.float32, name="strikes_before_pitch")

    pitcher_is_home = L.Input(shape=(1,), dtype=tf.float32, name="pitcher_is_home")
    pitcher_is_right = L.Input(shape=(1,), dtype=tf.float32, name="pitcher_is_right")
    batter_is_right = L.Input(shape=(1,), dtype=tf.float32, name="batter_is_right")

    # If you want to include pitch_num_in_pa:
    pitch_num_in_pa = L.Input(shape=(1,), dtype=tf.float32, name="pitch_num_in_pa")

    numeric = L.Concatenate(name="numeric_concat")([
        inning,
        balls,
        strikes,
        pitcher_is_home,
        pitcher_is_right,
        batter_is_right,
        pitch_num_in_pa,
    ])
    # No normalization - numeric features are already in reasonable ranges (0-10)

    # -------------------------
    # Categorical ID inputs
    # -------------------------
    pitcher_idx = L.Input(shape=(1,), dtype=tf.int32, name="pitcher_idx")
    batter_idx = L.Input(shape=(1,), dtype=tf.int32, name="batter_idx")

    pitcher_emb = L.Embedding(n_pitchers, pitcher_emb_dim, name="pitcher_emb")(pitcher_idx)
    batter_emb = L.Embedding(n_batters, batter_emb_dim, name="batter_emb")(batter_idx)

    pitcher_vec = L.Flatten()(pitcher_emb)
    batter_vec = L.Flatten()(batter_emb)

    # -------------------------
    # Lagged pitch features
    # -------------------------
    # pitch-1_idx, pitch-2_idx, pitch-3_idx
    p1 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_1_idx")
    p2 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_2_idx")
    p3 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_3_idx")
    p4 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_4_idx")
    p5 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_5_idx")
    p6 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_6_idx")
    p7 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_7_idx")
    p8 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_8_idx")
    p9 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_9_idx")
    p10 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_10_idx")

    # pitch-1_bucket_idx, ...
    b1 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_1_bucket_idx")
    b2 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_2_bucket_idx")
    b3 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_3_bucket_idx")

    # Embeddings for lagged pitch types (use vocab size to include index 0 for padding)
    p_emb_layer = L.Embedding(n_pitch_types_vocab, pitch_emb_dim, mask_zero=True, name="pitch_type_emb")
    b_emb_layer = L.Embedding(n_pitch_buckets_vocab, bucket_emb_dim, mask_zero=True, name="pitch_bucket_emb")

    p1v = L.Flatten()(p_emb_layer(p1))
    p2v = L.Flatten()(p_emb_layer(p2))
    p3v = L.Flatten()(p_emb_layer(p3))
    p4v = L.Flatten()(p_emb_layer(p4))
    p5v = L.Flatten()(p_emb_layer(p5))
    p6v = L.Flatten()(p_emb_layer(p6))
    p7v = L.Flatten()(p_emb_layer(p7))
    p8v = L.Flatten()(p_emb_layer(p8))
    p9v = L.Flatten()(p_emb_layer(p9))
    p10v = L.Flatten()(p_emb_layer(p10))

    b1v = L.Flatten()(b_emb_layer(b1))
    b2v = L.Flatten()(b_emb_layer(b2))
    b3v = L.Flatten()(b_emb_layer(b3))

    lag_vec = L.Concatenate(name="lag_concat")([p1v, p2v, p3v, p4v, p5v, p6v, p7v, p8v, p9v, p10v, b1v, b2v, b3v])

    # -------------------------
    # Combine all features
    # -------------------------
    x = L.Concatenate(name="all_features")([numeric, pitcher_vec, batter_vec, lag_vec])

    # Build variable number of hidden layers
    for i, units in enumerate(hidden_units):
        x = L.Dense(units, activation=None, name=f"dense_{i}")(x)
        x = L.BatchNormalization(name=f"bn_{i}")(x)
        x = L.Activation("relu", name=f"relu_{i}")(x)
        x = L.Dropout(dropout, name=f"dropout_{i}")(x)

    if predict == "pitch_type":
        # Output layer has n_pitch_types units (13), but we need to handle 1-indexed targets (1-13)
        # So we add an extra logit for index 0 (which should never be predicted)
        out_dim = n_pitch_types_vocab  # 14 units (0-13)
        y = L.Dense(out_dim, activation=None, name="pitch_type_logits")(x)
        # Use from_logits=True for numerical stability
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        # Same for buckets
        out_dim = n_pitch_buckets_vocab  # 4 units (0-3)
        y = L.Dense(out_dim, activation=None, name="pitch_bucket_logits")(x)
        # Use from_logits=True for numerical stability
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model = tf.keras.Model(
        inputs=[
            inning, balls, strikes,
            pitcher_is_home, pitcher_is_right, batter_is_right,
            pitch_num_in_pa,
            pitcher_idx, batter_idx,
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
            b1, b2, b3,
        ],
        outputs=y,
        name="pitch_sequence_ten_lags",
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,  # Much lower learning rate
        clipnorm=0.1  # Much more aggressive gradient clipping
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["sparse_categorical_accuracy"],
    )
    return model
