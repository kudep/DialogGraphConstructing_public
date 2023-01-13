import numpy as np
from collections import defaultdict
import typing as tp

from clustering.interface import OneViewClustering, Cluster
from dataset import Utterance, Dialogue


class SklearnClustering(OneViewClustering):
    def __init__(self, clustering, **config):
        self.clustering = clustering(**config)
        self.fitted = False
        self.labels_ = None

    def fit(self, embeddings: np.array) -> 'SklearnClustering':
        self.labels_ = self.clustering.fit_predict(embeddings)

        self.clusters_list = defaultdict(list)
        self.clusters = {}
        labels = self._get_original_labels()
        for idx, cluster in enumerate(labels):
            self.clusters_list[cluster].append(idx)
        for key in self.clusters_list:
            self.clusters[key] = Cluster(key, np.array(self.clusters_list[key]))

        self.fitted = True
        return self

    def get_cluster(self, idx) -> Cluster:
        assert self.fitted, "SklearnClustering must be fitted"
        return self.clusters[idx]

    def get_utterance_cluster(self, utterance_idx) -> Cluster:
        assert self.fitted, "SklearnClustering must be fitted"
        labels = self._get_original_labels()
        return self.clusters[labels[utterance_idx]]

    def get_nclusters(self) -> int:
        return self.clustering.n_clusters_

    def _get_original_labels(self):
        return self.labels_

    def predict_cluster(self, embedding: np.array,
                        utterance: tp.Optional[Utterance] = None,
                        dialogue: tp.Optional[Dialogue] = None):
        labels = self.clustering.predict(embedding[None, :])
        return self.get_cluster(labels[0])

    def get_labels(self) -> np.array:
        labels = self._get_original_labels()
        return np.array([self.clusters[l].id for l in labels])
