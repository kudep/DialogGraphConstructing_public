from re import I
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm, trange
from IPython.display import clear_output
from dataset import DialogueDataset, Dialogue, Utterance
from clustering.interface import OneViewClustering


def get_cluster_representation(dialogue: Dialogue,
                               dialogues: DialogueDataset,

                               clustering: OneViewClustering):
    bias = dialogues.get_dialog_start_idx(dialogue)
    encoded = []
    for i in range(len(dialogue)):
        encoded.append(clustering.get_utterance_cluster(bias + i).id)
    return np.array(encoded)


def get_new_cluster_representation(dialogue, embedder, clustering):
    embs = embedder.encode_new_dialogue(dialogue)
    if isinstance(embs, scipy.sparse.spmatrix):
        embs = embs.toarray()
    encoded = []
    for utt, emb in zip(dialogue, embs):
        encoded.append(clustering.predict_cluster(emb, utt, dialogue).id)
    return np.array(encoded)


def encode_train_with_clusters(dialogues, clustering):
    return [get_cluster_representation(dialogue, dialogues, clustering)
            for dialogue in tqdm(dialogues)]


def encode_test_with_clusters(dialogues, embedder, clustering):
    return [get_new_cluster_representation(dialogue, embedder, clustering)
            for dialogue in tqdm(dialogues)]


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def construct_dataset(clusters_data, n_neighbours=2, n_clusters=200):
    x = []
    y = []
    for dialogue in clusters_data:
        for center in range(len(dialogue)):
            ohe = np.zeros(n_clusters)
            for i in range(max(0, center - n_neighbours),
                           min(center + n_neighbours + 1, len(dialogue))):
                if i != center:
                    ohe[dialogue[i]] = 1
            x.append(dialogue[center])
            y.append(ohe)
    return x, np.stack(y, axis=0)


def get_w2v_model(n_tokens, hid_size=32):
    return nn.Sequential(
        nn.Embedding(n_tokens, hid_size),
        nn.Linear(hid_size, n_tokens)
    )


def vtqdm(iterable, verbosity=1, *args, **kwargs):
    if verbosity:
        return tqdm(iterable, *args, **kwargs)
    else:
        return iterable


def train_model(model, optimizer, criterion, log, train_loader, val_loader=None,
                n_epochs=40, plot_every=2, patience=3, continue_from=0, verbosity=0):
    stop_count = 0

    for epoch in vtqdm(range(continue_from, n_epochs), verbosity=verbosity, leave=False):
        train_losses = []
        for x, y in vtqdm(train_loader, verbosity=verbosity, leave=False):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        log['train_loss'].append(np.mean(train_losses))

        if val_loader is not None:
            with torch.no_grad():
                losses = []
                for x, y in vtqdm(val_loader, verbosity=verbosity, leave=False):
                    x = x.to(device)
                    y = y.to(device)

                    preds = model(x)
                    loss = criterion(preds, y)
                    losses.append(loss.item())
                log['val_loss'].append(np.mean(losses))

        if verbosity > 1 and epoch % plot_every == 0:
            clear_output(True)
            plt.figure(figsize=(12, 8))
            plt.grid(True)
            plt.plot(log['train_loss'], label='train')
            plt.plot(log['val_loss'], label='val')
            plt.title('Cluster2Vec training')
            plt.show()

        if len(log['val_loss']) > 1 and \
                log['val_loss'][-1] > log['val_loss'][-2]:
            stop_count += 1
            if stop_count >= patience:
                break
        else:
            stop_count = 0


def train_cluster_embeddings(train, embedder, clustering, val=None, n_neighbours=2,
                             batch_size=64, device=device, hidden_size=8, verbosity=0,
                             **train_kwargs):
    train_clusters = encode_train_with_clusters(train, clustering)
    X_train, y_train = construct_dataset(train_clusters,
                                         n_neighbours=n_neighbours,
                                         n_clusters=clustering.get_nclusters())
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)

    if val is not None:
        val_clusters = encode_test_with_clusters(val, embedder, clustering)
        X_val, y_val = construct_dataset(val_clusters, n_neighbours=n_neighbours,
                                        n_clusters=clustering.get_nclusters())
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    model = get_w2v_model(clustering.get_nclusters(), hid_size=hidden_size).to(
        device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    log = {
        'train_loss': [],
        'val_loss': []
    }

    train_model(model, optimizer, criterion, log, train_loader, val_loader=val_loader,
                verbosity=verbosity, **train_kwargs)

    cluster_embeddings = list(model.children())[0].weight.detach().cpu().numpy()

    return cluster_embeddings
