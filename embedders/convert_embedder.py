import numpy as np

import typing as tp

from dataset import DialogueDataset, Dialogue
from embedders import OneViewEmbedder
from embedders.convert import encode_context, encode_responses


class ConvertEmbedder(OneViewEmbedder):
    def encode_dialogue(self, dialogue: Dialogue) -> np.array:
        utterances = [utt.utterance for utt in dialogue]
        embeddings = np.array([encode_context(utterances[:k]) for k in
                               range(1, len(utterances) + 1)])
        return embeddings

    def encode_new_dialogue(self, dialogue: Dialogue):
        return self.encode_dialogue(dialogue)

    def encode_responses(self, utterances: tp.List[str]):
        return encode_responses(utterances)
