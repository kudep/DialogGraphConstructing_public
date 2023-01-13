import numpy as np

from dataset import DialogueDataset
from dataset.multiwoz import load_multiwoz
from embedders import SentenceEmbedder, CachedEmbeddings
from clustering import SubClustering, KMeansClustering
from clustering.filters import speaker_filter
from clustering.metrics import score_clustering
from dialogue_graph import FrequencyDialogueGraph
from cluster2vec import full_merge_build


def dgac_one_stage(train, n_clusters=60, verbosity=0):
    embedder = SentenceEmbedder(device=0)
    train_embeddings = embedder.encode_dataset(train)

    if verbosity > 0:
        print('Clustering...')
    clustering = SubClustering(train, KMeansClustering, speaker_filter,
                               {'n_clusters': n_clusters // 2}).fit(train_embeddings)
    if verbosity > 0:
        print(f'Clustering done! Total {clustering.get_nclusters()} clusters.')

    if verbosity > 0:
        print('Building graph...')
    graph = FrequencyDialogueGraph(train, embedder, clustering)
    graph.build()

    return graph


def dgac_two_stage(train, n_clusters=60, val=None, n_clusters_first_stage=400, verbosity=0):
    embedder = SentenceEmbedder(device=0)
    train_embeddings = embedder.encode_dataset(train)

    if verbosity > 0:
        print('Clustering...')
    clustering = SubClustering(train, KMeansClustering, speaker_filter,
                                        {'n_clusters': n_clusters_first_stage // 2}).fit(train_embeddings)
    if verbosity > 0:
        print(f'Clustering done! Total {clustering.get_nclusters()} clusters.')

    if verbosity > 0:
        print('Building graph...')
    (
        graph_merged,
        cluster_embeddings,
        cluster_kmeans_labels,
        clustering_merged
    ) = full_merge_build(train, embedder, clustering,
                         val=val, n_clusters=n_clusters, verbosity=verbosity)

    return graph_merged
