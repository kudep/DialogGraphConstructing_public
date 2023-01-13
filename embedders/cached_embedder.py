import numpy as np

from dataset import DialogueDataset, Dialogue
from embedders import OneViewEmbedder


class CachedEmbeddings(OneViewEmbedder):
    def __init__(self, dialogues: DialogueDataset, embeddings: np.array,
                 test_dialogues: DialogueDataset = None,
                 test_embeddings: np.array = None):
        super().__init__()
        self.dialogues = dialogues
        self.embeddings = embeddings
        self.test_dialogues = test_dialogues
        self.test_embeddings = test_embeddings

    def encode_dialogue(self, dialogue: Dialogue) -> np.array:
        idx = self.dialogues.get_dialog_start_idx(dialogue)
        return self.embeddings[idx:idx + len(dialogue)]

    def encode_new_dialogue(self, dialogue: Dialogue):
        idx = self.test_dialogues.get_dialog_start_idx(dialogue)
        return self.test_embeddings[idx:idx + len(dialogue)]

    def encode_utterances(self, utts):
        return self.embeddings[utts]
