import tensorflow as tf
from tensorflow.keras import layers as L

def build_pitch_model(
    n_pitchers: int,
    n_batters: int,
    n_pitch_types: int,       # if predicting pitch_idx
    n_pitch_buckets: int,     # if predicting pitch_bucket_idx
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
    numeric = L.LayerNormalization()(numeric)

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

    # pitch-1_bucket_idx, ...
    b1 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_1_bucket_idx")
    b2 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_2_bucket_idx")
    b3 = L.Input(shape=(1,), dtype=tf.int32, name="pitch_3_bucket_idx")

    # Embeddings for lagged pitch types
    p_emb_layer = L.Embedding(n_pitch_types, pitch_emb_dim, name="pitch_type_emb")
    b_emb_layer = L.Embedding(n_pitch_buckets, bucket_emb_dim, name="pitch_bucket_emb")

    p1v = L.Flatten()(p_emb_layer(p1))
    p2v = L.Flatten()(p_emb_layer(p2))
    p3v = L.Flatten()(p_emb_layer(p3))

    b1v = L.Flatten()(b_emb_layer(b1))
    b2v = L.Flatten()(b_emb_layer(b2))
    b3v = L.Flatten()(b_emb_layer(b3))

    lag_vec = L.Concatenate(name="lag_concat")([p1v, p2v, p3v, b1v, b2v, b3v])

    # -------------------------
    # Combine all features
    # -------------------------
    x = L.Concatenate(name="all_features")([numeric, pitcher_vec, batter_vec, lag_vec])
    x = L.Dense(hidden_units[0], activation="relu")(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(hidden_units[1], activation="relu")(x)
    x = L.Dropout(dropout)(x)

    if predict == "pitch_type":
        out_dim = n_pitch_types
        y = L.Dense(out_dim, activation="softmax", name="pitch_type_probs")(x)
        loss = "sparse_categorical_crossentropy"
    else:
        out_dim = n_pitch_buckets
        y = L.Dense(out_dim, activation="softmax", name="pitch_bucket_probs")(x)
        loss = "sparse_categorical_crossentropy"

    model = tf.keras.Model(
        inputs=[
            inning, balls, strikes,
            pitcher_is_home, pitcher_is_right, batter_is_right,
            pitch_num_in_pa,
            pitcher_idx, batter_idx,
            p1, p2, p3,
            b1, b2, b3,
        ],
        outputs=y,
        name="pitch_sequence_baseline",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=["sparse_categorical_accuracy"],
    )
    return model
