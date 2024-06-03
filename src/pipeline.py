from typing import Sequence

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import Sequential

from embeddings import BaseEmbedding


class Model:
    def __init__(self, emb_pipeline: Sequence[BaseEmbedding]):
        self.emb_pipeline = emb_pipeline

        # TODO: Add model builder
        self.model = Sequential(
            [
                InputLayer(shape=(6539,)),
                Dropout(0.25),
                Dense(512, activation="relu"),
                Dropout(0.25),
                Dense(512, activation="relu"),
                Dropout(0.25),
                Dense(256, activation="relu"),
                Dropout(0.25),
                Dense(128, activation="relu"),
                Dropout(0.25),
                Dense(64, activation="relu"),
                Dropout(0.25),
                Dense(1, activation="sigmoid"),
            ]
        )

        # TODO: Rewrite to ranking problem
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.emb_pipeline.fit(x.values, y.values)
        x_transformed = self.emb_pipeline.transform(x.values)

        _ = self.model.fit(x_transformed, y.values, epochs=50, batch_size=32, verbose=1)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        pass
