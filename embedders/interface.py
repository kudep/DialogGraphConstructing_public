import numpy as np
from abc import ABC, abstractmethod
from tqdm.notebook import tqdm

import typing as tp

from dataset import DialogueDataset, Dialogue


class OneViewEmbedder(ABC):
    def __init__(self, config: tp.Optional[tp.Any] = None):
        self.config = config

    @abstractmethod
    def encode_dialogue(self, dialogue: Dialogue):
        return np.zeros((len(dialogue), 1), dtype=np.int32)

    def encode_dataset(self, dialogues: DialogueDataset):
        return np.concatenate(
            [self.encode_dialogue(dialogue) for dialogue in tqdm(dialogues)],
            axis=0)

    def encode_new_dialogue(self, dialogue: Dialogue):
        return self.encode_dialogue(dialogue)

    def encode_new_dataset(self, dialogues: DialogueDataset):
        return np.concatenate(
            [self.encode_new_dialogue(dialogue) for dialogue in
             tqdm(dialogues)], axis=0)
