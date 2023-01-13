from sklearn.metrics import silhouette_score, calinski_harabasz_score, \
    davies_bouldin_score


def score_clustering(clustering, embeddings, each=1):
    embeddings = embeddings[::each]
    labels = clustering.get_labels()[::each]
    sil = silhouette_score(embeddings, labels)
    ch = calinski_harabasz_score(embeddings, labels)
    db = davies_bouldin_score(embeddings, labels)
    return sil, ch, db
