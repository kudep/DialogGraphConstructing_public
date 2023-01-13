import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def merge_clusters(cluster_embeddings, n_clusters=15):
    cluster_kmeans = KMeans(n_clusters=n_clusters).fit(cluster_embeddings)
    return cluster_kmeans.labels_


def merge_clusters_with_sep(cluster_embeddings, separator, clustering,
                            dialogues, n_clusters=15):
    groups = defaultdict(list)
    for cluster in range(clustering.get_nclusters()):
        utt_idx = clustering.get_cluster(cluster).utterances[0]
        utt = dialogues.get_utterance_by_idx(utt_idx)
        dialogue = dialogues.get_dialogue_by_idx(utt_idx)
        groups[separator(utt, dialogue)].append(
            clustering.get_cluster(cluster).id)

    labels = np.zeros(clustering.get_nclusters(), dtype=int)
    shift = 0
    for group, clusters in groups.items():
        local_embeddings = cluster_embeddings[clusters]
        cluster_kmeans = KMeans(n_clusters=n_clusters).fit(local_embeddings)
        for cluster_id, local_lab in zip(clusters, cluster_kmeans.labels_):
            labels[cluster_id] = local_lab + shift
        shift += n_clusters

    return labels


def separate_clusters(cluster_labels, separator, clustering, dialogues):
    groups = {}
    for cluster in range(clustering.get_nclusters()):
        utt_idx = clustering.get_cluster(cluster).utterances[0]
        cluster_id = clustering.get_cluster(cluster).id
        cluster_label = cluster_labels[cluster_id]
        utt = dialogues.get_utterance_by_idx(utt_idx)
        dialogue = dialogues.get_dialogue_by_idx(utt_idx)
        if cluster_label not in groups:
            groups[cluster_label] = defaultdict(list)
        groups[cluster_label][separator(utt, dialogue)].append(cluster_id)

    new_labels = np.zeros_like(cluster_labels)
    cluster_idx = 0
    for dd in groups.values():
        for cluster in dd.values():
            for item in cluster:
                new_labels[item] = cluster_idx
            cluster_idx += 1
    return new_labels


def plot_merge_clustering(cluster_embeddings, merge_labels):
    clusters_tsne = TSNE().fit_transform(cluster_embeddings)

    plt.figure(figsize=(15, 10))

    x, y = clusters_tsne.T
    plt.scatter(x, y, c=merge_labels)

    plt.show()
