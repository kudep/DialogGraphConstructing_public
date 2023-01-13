import numpy as np
import typing as tp
from collections import Counter
from itertools import accumulate

from sklearn.cluster import SpectralCoclustering

from clustering.interface import Cluster, OneViewClustering
from dataset import Dialogue, Utterance


class BiCluster(Cluster):
    def __init__(self, cluster_id, utterances, words):
        super().__init__(cluster_id, utterances)
        self.words = words


class BiClustering(OneViewClustering):
    def __init__(self, n_clusters=15, svd_method="arpack", random_state=42,
                 **config):
        self.n_clusters = n_clusters
        self.svd_method = svd_method
        self.random_state = random_state
        self.config = config

        self.cocluster = SpectralCoclustering(
            n_clusters=self.n_clusters,
            svd_method=self.svd_method,
            random_state=self.random_state,
            **config
        )
        self.fitted = False

    def fit(self, embeddings: np.array) -> 'BiClustering':
        mask = np.array(embeddings.sum(axis=1) > 0).reshape(-1)

        word_mask = np.array(embeddings[mask].sum(axis=0) > 0).reshape(-1)

        self.filtered_mapping = {}
        for orig, mapped in enumerate(accumulate(mask.astype(np.int32))):
            if mapped > 0 and mapped - 1 not in self.filtered_mapping:
                self.filtered_mapping[mapped - 1] = orig

        self.words_mapping = {}
        for orig, mapped in enumerate(accumulate(word_mask.astype(np.int32))):
            if mapped > 0 and mapped - 1 not in self.words_mapping:
                self.words_mapping[mapped - 1] = orig

        self.cocluster.fit(embeddings[mask].T[word_mask].T)

        self.cluster_by_utt = {}
        self.clusters = []
        for cluster_id in range(self.n_clusters):
            utts, words = self.cocluster.get_indices(cluster_id)
            utts = np.array([self.filtered_mapping[utt] for utt in utts])
            words = np.array([self.words_mapping[word] for word in words])
            self.clusters.append(BiCluster(cluster_id, utts, words))

            for utt in utts:
                self.cluster_by_utt[utt] = cluster_id

        self.fitted = True
        return self

    def get_cluster(self, idx) -> BiCluster:
        assert self.fitted, "BiClustering must be fitted"
        return self.clusters[idx]

    def get_utterance_cluster(self, utterance_idx) -> BiCluster:
        assert self.fitted, "BiClustering must be fitted"
        return self.clusters[self.cluster_by_utt[utterance_idx]]

    def get_nclusters(self) -> int:
        return self.n_clusters

    def predict_cluster(self, embedding: np.array,
                        utterance: tp.Optional[Utterance] = None,
                        dialogue: tp.Optional[Dialogue] = None):
        raise NotImplementedError

    def get_labels(self) -> np.array:
        return np.array(self.get_utterance_cluster(idx).id
                        for idx in range(len(self.cluster_by_utt)))


def print_biclustering(clustering, embedder, max_utts=10, max_words=50):
    for i in range(clustering.get_nclusters()):
        cluster = clustering.get_cluster(i)
        utts = cluster.utterances
        words = cluster.words

        print(f"Cluster #{i}: {len(utts)} utterances, {len(words)} words")
        utt_step = (len(utts) + max_utts - 1) // max_utts
        word_step = (len(words) + max_words - 1) // max_words
        if word_step > 0:
            print(
                '; '.join(embedder.tfidf.get_feature_names()[word] for word in
                          words[::word_step]))
        else:
            print('NO WORDS')
        print()
        if utt_step > 0:
            print('\n'.join(embedder.dialogues.utterances[utt] for utt in
                            utts[::utt_step]))
            print()
            print(Counter(
                [embedder.dialogues.get_utterance_by_idx(utt).speaker for utt
                 in utts]))
        else:
            print('NO UTTERANCES')
        print('\n')
