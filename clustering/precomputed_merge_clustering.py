from copy import deepcopy
from collections import defaultdict
import numpy as np
import typing as tp
from clustering.interface import Cluster, OneViewClustering
from dataset import Dialogue, Utterance, DialogueDataset


class MergeCluster(Cluster):
    def __init__(self, cluster_id, clusters):
        utterances = sum([cluster.utterances for cluster in clusters], [])
        super().__init__(cluster_id, utterances)
        self.clusters = clusters


class PrecomputedMergeClustering(OneViewClustering):
    def __init__(self, dialogues: DialogueDataset,
                 subclustering: OneViewClustering,
                 merge_clustering_labels: np.ndarray):
        self.dialogues = dialogues
        self._subclustering = subclustering
        self._merge_clustering = merge_clustering_labels
        self.n_clusters = len(np.unique(merge_clustering_labels))

        self._by_clusters = defaultdict(list)
        for cluster in range(self._subclustering.get_nclusters()):
            label = merge_clustering_labels[
                self._subclustering.get_cluster(cluster).id]
            self._by_clusters[label].append(cluster)

        self.clusters = {}
        for label, cluster_ids in self._by_clusters.items():
            clusters = [self._subclustering.get_cluster(idx) for idx in
                        cluster_ids]
            self.clusters[label] = MergeCluster(label, clusters)

        self.fitted = True

    def fit(self, embeddings: np.array) -> 'PrecomputedMergeClustering':
        pass

    def get_cluster(self, idx: int) -> Cluster:
        assert self.fitted, "Clustering must be fitted"

        return self.clusters[idx]

    def get_utterance_cluster(self, utterance_idx) -> Cluster:
        assert self.fitted, "Clustering must be fitted"

        cluster_id = self._subclustering.get_utterance_cluster(
            utterance_idx).id

        return self.clusters[self._merge_clustering[cluster_id]]

    def get_nclusters(self) -> int:
        return self.n_clusters

    def predict_cluster(self, embedding: np.array,
                        utterance: tp.Optional[Utterance] = None,
                        dialogue: tp.Optional[Dialogue] = None):
        assert utterance is not None and dialogue is not None, \
            "Utterance and dialogue must be set for subclustering predictions"
        cluster_id = self._subclustering.predict_cluster(embedding, utterance,
                                                         dialogue).id
        return self.clusters[self._merge_clustering[cluster_id]]

    def get_labels(self) -> np.array:
        labels = deepcopy(self._subclustering.get_labels())
        labels = np.array(
            list(map(lambda x: self._merge_clustering[int(x)], labels)),
            dtype=int)
        return labels
