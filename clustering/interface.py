from abc import ABC, abstractmethod
import numpy as np

import typing as tp

from dataset import Utterance, Dialogue


class Cluster:
    def __init__(self, cluster_id, utterances):
        self.id = cluster_id
        self.utterances = utterances

    def __getitem__(self, idx):
        return self.utterances[idx]

    def __iter__(self):
        return iter(self.utterances)

    def __len__(self):
        return len(self.utterances)


class OneViewClustering(ABC):
    def __init__(self):
        self.size = 1
        self.cluster = None

    @abstractmethod
    def fit(self, embeddings: np.array) -> 'OneViewClustering':
        self.size = embeddings.shape[0]
        self.cluster = Cluster(0, np.arange(self.size))
        return self

    @abstractmethod
    def get_cluster(self, idx: int) -> Cluster:
        assert idx == 1
        return self.cluster

    @abstractmethod
    def get_utterance_cluster(self, utt_idx: int) -> Cluster:
        return self.cluster

    @abstractmethod
    def get_nclusters(self) -> int:
        return 1

    @abstractmethod
    def predict_cluster(self, embedding: np.array,
                        utterance: tp.Optional[Utterance] = None,
                        dialogue: tp.Optional[Dialogue] = None):
        return self.cluster

    @abstractmethod
    def get_labels(self) -> np.array:
        return np.zeros(self.size)
