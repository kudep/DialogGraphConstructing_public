from sklearn.cluster import KMeans

from clustering.sklearn_clustering import SklearnClustering


class KMeansClustering(SklearnClustering):
    def __init__(self, n_clusters=15, random_state=42, **config):
        self.n_clusters = n_clusters
        super().__init__(KMeans, n_clusters=n_clusters,
                         random_state=random_state, **config)

    def get_nclusters(self) -> int:
        return self.n_clusters

    def _get_original_labels(self):
        return self.clustering.labels_
