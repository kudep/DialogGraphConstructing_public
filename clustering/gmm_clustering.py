from sklearn.mixture import GaussianMixture

from clustering.sklearn_clustering import SklearnClustering


class GMMClustering(SklearnClustering):
    def __init__(self, n_clusters=15, random_state=42, **config):
        self.n_clusters = n_clusters
        self._labels = None
        super().__init__(GaussianMixture, n_components=n_clusters,
                         random_state=random_state, **config)

    def get_nclusters(self) -> int:
        return self.n_clusters
