from abc import ABC
from abc import abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class BaseEmbedding(ABC):
    @abstractmethod
    def fit(self, x: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    @abstractmethod
    def transform(self, x: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class EmbeddingPipeline(BaseEmbedding):
    def __init__(self, embs: Sequence[BaseEmbedding]):
        self.embs = embs

    def fit(self, x: pd.DataFrame, y: pd.Series):
        for emb in self.embs:
            emb.fit(x=x, y=y)

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        transformation_list = []
        for emb in self.embs:
            transformation_list.append(emb.transform(x))
        return np.concatenate(transformation_list, axis=1)


class SimilarityEmbedding(BaseEmbedding):
    def __init__(self):
        self.features = None
        self.item_indices = {}

    def fit(self, x: pd.DataFrame, y: pd.Series):
        x_pivot = x.pivot_table(index=["user_id"], columns=["item_id"], values="response")
        x_pivot = x_pivot.fillna(x_pivot.mean())
        self.features = cosine_similarity(x_pivot.values.T)

        items, indices = np.unique(x["item_id"], return_index=True)
        self.item_indices = {item: idx for item, idx in zip(items.tolist(), indices.tolist())}

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        indices = x["item_id"].replace(self.item_indices)
        return self.features[indices]
