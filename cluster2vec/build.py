from clustering.precomputed_merge_clustering import PrecomputedMergeClustering
from dialogue_graph import FrequencyDialogueGraph
from .embeddings import train_cluster_embeddings
from .merge_clusters import merge_clusters_with_sep, merge_clusters, \
    separate_clusters, plot_merge_clustering


def build_graph(train, embedder, clustering, merge_labels):
    merged_clustering = PrecomputedMergeClustering(train, clustering, merge_labels)
    graph_merged = FrequencyDialogueGraph(train, embedder, merged_clustering)
    graph_merged.build()
    return graph_merged, merged_clustering


def full_merge_build(train, embedder, clustering, val=None, n_clusters=15,
                     separator=None,
                     sep_before=True, n_neighbours=2, hidden_size=8,
                     n_epochs=40, verbosity=0):
    cluster_embeddings = train_cluster_embeddings(train, embedder,
                                                  clustering, val=val,
                                                  n_neighbours=n_neighbours,
                                                  hidden_size=hidden_size,
                                                  n_epochs=n_epochs, verbosity=verbosity)
    if separator is not None and sep_before:
        cluster_kmeans_labels = merge_clusters_with_sep(cluster_embeddings,
                                                        separator, clustering,
                                                        train,
                                                        n_clusters=n_clusters // 2)
    else:
        cluster_kmeans_labels = merge_clusters(cluster_embeddings,
                                               n_clusters=n_clusters)
        if separator is not None:
            cluster_kmeans_labels = separate_clusters(cluster_kmeans_labels,
                                                      separator, clustering,
                                                      train)
    plot_merge_clustering(cluster_embeddings, cluster_kmeans_labels)
    graph, clustering = build_graph(train, embedder,
                                clustering, cluster_kmeans_labels)
    return graph, cluster_embeddings, cluster_kmeans_labels, clustering
