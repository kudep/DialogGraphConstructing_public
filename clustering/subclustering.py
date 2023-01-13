import numpy as np
from collections import defaultdict

import typing as tp

from clustering.interface import OneViewClustering, Cluster
from dataset import Dialogue, DialogueDataset, Utterance


class SubClustering(OneViewClustering):
    def __init__(self, dialogues: DialogueDataset,
                 subclustering: OneViewClustering,
                 separator: tp.Optional[
                     tp.Callable[[Utterance, Dialogue], tp.Hashable]],
                 clustering_config: tp.Dict[str, tp.Any]):
        self.dialogues = dialogues
        self._clustering_config = clustering_config
        self._subclustering = subclustering
        self._separator = separator
        self.fitted = False

        self._split_dataset()

    def _split_dataset(self):
        self._groups = defaultdict(list)
        self._group_by_idx = {}
        self._idx_in_group = []
        for utt_idx in range(len(self.dialogues.utterances)):
            dialogue = self.dialogues.get_dialogue_by_idx(utt_idx)
            utt = self.dialogues.get_utterance_by_idx(utt_idx)
            group = self._separator(utt, dialogue)
            self._idx_in_group.append(len(self._groups[group]))
            self._groups[group].append(utt_idx)
            self._group_by_idx[utt_idx] = group

    def fit(self, embeddings: np.array) -> 'SubClustering':
        self._subclusters = {}
        self._cluster_gid = []
        self._cluster_idx_in_group = []

        for gid, group in self._groups.items():
            group = np.array(group)
            self._subclusters[gid] = self._subclustering(
                **self._clustering_config).fit(embeddings[group])
            for idx in range(self._subclusters[gid].get_nclusters()):
                cluster = self._subclusters[gid].get_cluster(idx)
                cluster.id = len(self._cluster_gid)
                cluster.utterances = [self._groups[gid][utt_idx] for utt_idx in
                                      cluster.utterances]
                self._cluster_gid.append(gid)
                self._cluster_idx_in_group.append(idx)

        self.fitted = True
        return self

    def get_cluster(self, idx: int) -> Cluster:
        assert self.fitted, "Clustering must be fitted"

        gid = self._cluster_gid[idx]
        group_idx = self._cluster_idx_in_group[idx]
        cluster = self._subclusters[gid].get_cluster(group_idx)
        return cluster

    def get_utterance_cluster(self, utterance_idx) -> Cluster:
        assert self.fitted, "Clustering must be fitted"

        group = self._group_by_idx[utterance_idx]
        group_uidx = self._idx_in_group[utterance_idx]
        return self._subclusters[group].get_utterance_cluster(group_uidx)

    def get_nclusters(self) -> int:
        return len(self._cluster_gid)

    def predict_cluster(self, embedding: np.array,
                        utterance: tp.Optional[Utterance],
                        dialogue: tp.Optional[Dialogue]):
        assert utterance is not None and dialogue is not None, \
            "Utterance and dialogue must be set for subclustering predictions"
        group = self._separator(utterance, dialogue)
        subcluster = self._subclusters[group]
        return subcluster.predict_cluster(embedding, utterance, dialogue)

    def get_labels(self) -> np.array:
        labels = np.zeros(len(self.dialogues.utterances))
        for gid, group in self._groups.items():
            group = np.array(group)
            labels[group] = self._subclusters[gid].get_labels()
        return labels
