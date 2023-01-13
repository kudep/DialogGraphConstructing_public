import numpy as np
import scipy.sparse

from copy import deepcopy

from dataset import DialogueDataset, Dialogue
from embedders.interface import OneViewEmbedder
from clustering.interface import OneViewClustering


class FrequencyDialogueGraph:
    def __init__(self, dialogues: DialogueDataset, embedder: OneViewEmbedder,
                 clustering: OneViewClustering):
        self.dialogues: DialogueDataset = dialogues
        self.clustering: OneViewClustering = clustering
        self.embedder: OneViewEmbedder = embedder

        self.n_vertices = clustering.get_nclusters() + 1
        self.start_node = self.n_vertices - 1

        self.edges = [[0] * self.n_vertices for _ in range(self.n_vertices)]

        self.eps = 1e-5

    #         self._build()

    def _add_dialogue(self, dialogue: Dialogue) -> None:
        utt_idx = self.dialogues.get_dialog_start_idx(dialogue)
        current_node = self.start_node
        for utt in dialogue:
            next_node = self.clustering.get_utterance_cluster(utt_idx).id
            self.edges[current_node][next_node] += 1
            current_node = next_node
            utt_idx += 1

    def build(self):
        for dialogue in self.dialogues:
            self._add_dialogue(dialogue)

        self.probabilities = [np.array(self.edges[v]) / np.sum(self.edges[v])
                              for v in range(self.n_vertices)]

    def iter_dialogue(self, dialogue: Dialogue):
        d_embs = self.embedder.encode_new_dialogue(dialogue)
        if isinstance(d_embs, scipy.sparse.spmatrix):
            d_embs = d_embs.toarray()
        for utt, emb in zip(dialogue, d_embs):
            next_node = self.clustering.predict_cluster(emb, utt, dialogue).id
            yield next_node, emb
    
    def get_dataset_markup(self, dataset: DialogueDataset):
        labels = []
        for dialogue in dataset:
            for next_node, _ in self.iter_dialogue(dialogue):
                labels.append(next_node)
        return labels
    
    def get_transitions(self):
        return deepcopy(self.probabilities)
            

    def _dialogue_success_rate(self, dialogue: Dialogue, cluster_frequencies,
                               acc_ks=None):
        if acc_ks is None:
            acc_ks = []
        acc_ks = np.array(acc_ks)

        d_embs = self.embedder.encode_new_dialogue(dialogue)
        if isinstance(d_embs, scipy.sparse.spmatrix):
            d_embs = d_embs.toarray()

        logprob = 0
        accuracies = np.zeros(len(acc_ks))

        visited_clusters = []

        current_node = self.start_node
        for utt, emb in zip(dialogue, d_embs):
            next_node = self.clustering.predict_cluster(emb, utt, dialogue).id
            cluster_frequencies[next_node] += 1
            visited_clusters.append(next_node)
            prob = self.probabilities[current_node][next_node]
            prob = max(prob, self.eps)
            logprob -= np.log(prob) * prob

            next_cluster_ind = (self.probabilities[current_node] >= prob).sum()
            accuracies = accuracies + (next_cluster_ind <= acc_ks)

            current_node = next_node
        accuracies /= len(dialogue)
        unique_score = len(np.unique(visited_clusters)) / len(visited_clusters)
        return logprob, unique_score, accuracies

    def success_rate(self, test: DialogueDataset, acc_ks=None):
        if acc_ks is None:
            acc_ks = []
        logprob = 0
        accuracies = np.zeros(len(acc_ks))
        cluster_frequencies = np.zeros(self.n_vertices - 1)
        unique_score = 0.
        for dialogue in test:
            lp, us, acc = self._dialogue_success_rate(dialogue,
                                                      cluster_frequencies,
                                                      acc_ks=acc_ks)
            logprob += lp
            accuracies += acc
            unique_score += us
        logprob /= len(test)
        accuracies /= len(test)
        unique_score /= len(test)
        return logprob, cluster_frequencies, unique_score, accuracies
