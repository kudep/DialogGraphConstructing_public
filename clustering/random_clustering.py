import numpy as np
import typing as tp

from clustering.interface import OneViewClustering, Cluster
from dataset import Dialogue, Utterance


class RandomClustering(OneViewClustering):
    def __init__(self, n_clusters=15):
        super().__init__()
        self.n_clusters = n_clusters

    def fit(self, embeddings: np.array) -> 'KMeansClustering':
        self.clusters = [[] for _ in range(self.n_clusters)]
        self.embeddings = embeddings

        for utt_idx, emb in enumerate(embeddings):
            cluster = self._get_cluster_by_emb(emb)
            self.clusters[cluster].append(utt_idx)

        for key in range(self.n_clusters):
            self.clusters[key] = Cluster(key, np.array(self.clusters[key]))

        self.fitted = True
        return self

    def get_cluster(self, idx) -> Cluster:
        assert self.fitted, "RandomClustering must be fitted"
        return self.clusters[idx]

    def _get_cluster_by_emb(self, emb: np.array):
        return hash(tuple(emb)) % self.n_clusters

    def get_nclusters(self) -> int:
        return self.n_clusters

    def predict_cluster(self, embedding: np.array,
                        utterance: tp.Optional[Utterance] = None,
                        dialogue: tp.Optional[Dialogue] = None):
        return self.clusters[self._get_cluster_by_emb(embedding)]

    def get_utterance_cluster(self, utterance_idx) -> Cluster:
        assert self.fitted, "RandomClustering must be fitted"
        return self.clusters[
            self._get_cluster_by_emb(self.embeddings[utterance_idx])]

    def get_labels(self) -> np.array:
        return np.array(self.get_utterance_cluster(idx).id
                        for idx in range(len(self.embeddings)))
