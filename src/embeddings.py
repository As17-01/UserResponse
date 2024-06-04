from abc import ABC
from abc import abstractmethod
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


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
    def __init__(self, n_components: int = 100):
        self.features = None
        self.item_indices = {}
        self.pca = PCA(n_components=n_components)

    def fit(self, x: pd.DataFrame, y: pd.Series):
        # TODO: refactor
        x = x.copy()
        x["response"] = y

        x_pivot = x.pivot_table(index=["user_id"], columns=["item_id"], values="response")
        x_pivot = x_pivot.fillna(x_pivot.mean())
        self.features = cosine_similarity(x_pivot.values.T)

        items = np.unique(x["item_id"])
        self.item_indices = {item: idx for item, idx in zip(items.tolist(), np.arange(len(items)).tolist())}

        self.pca.fit(self.features)

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        indices = x["item_id"].copy().replace(self.item_indices)
        compressed_features = self.pca.transform(self.features)
        return compressed_features[indices]
