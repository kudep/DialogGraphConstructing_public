from sklearn.cluster import AgglomerativeClustering

from clustering.sklearn_clustering import SklearnClustering


class HierarchicalClustering(SklearnClustering):
    def __init__(self, n_clusters=15, **config):
        super().__init__(AgglomerativeClustering, n_clusters=n_clusters,
                         **config)
