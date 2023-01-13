import scipy.sparse
import typing as tp

from sklearn.feature_extraction.text import TfidfVectorizer

from embedders import OneViewEmbedder
from dataset import DialogueDataset, Dialogue


class TFIDFEmbedder(OneViewEmbedder):
    def __init__(self, **config: tp.Any):
        self.config = config
        self.embeddings = None

    def fit(self, dialogues: DialogueDataset):
        self.tfidf = TfidfVectorizer(**self.config)
        self.dialogues = dialogues
        self.embeddings = self.tfidf.fit_transform(self.dialogues.utterances)

    def encode_dialogue(self, dialogue: Dialogue) -> scipy.sparse.csr_matrix:
        assert self.embeddings is not None, "TFIDFEmbedder must be fitted before encode"
        d_start = self.dialogues.get_dialog_start_idx(dialogue)
        return self.embeddings[d_start: d_start + len(dialogue)]

    def encode_dataset(self,
                       dialogues: DialogueDataset) -> scipy.sparse.csr_matrix:
        assert self.embeddings is not None, "TFIDFEmbedder must be fitted before encode"
        return self.embeddings

    def get_utterance_keywords(self, idx):
        assert self.embeddings is not None, "TFIDFEmbedder must be fitted"
        feats = self.embeddings[idx].nonzero()[1]
        return [self.tfidf.get_feature_names()[feat] for feat in feats]

    def encode_new_dialogue(self, dialogue: Dialogue):
        new_embeddings = self.tfidf.transform(
            [utt.utterance for utt in dialogue])
        return new_embeddings

    def encode_new_dataset(self, dialogues: DialogueDataset):
        new_embeddings = self.tfidf.transform(dialogues.utterances)
        return new_embeddings
