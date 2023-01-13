import numpy as np
import typing as tp

from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import trange

from embedders import OneViewEmbedder
from dataset import DialogueDataset, Dialogue


class BertEmbedder(OneViewEmbedder):
    def __init__(self, model_name: str = "bert-base-uncased", device='cpu',
                 batch_size=128, checkpoint_path=None):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name).to(device)
        if checkpoint_path is not None:
            kw = torch.load(checkpoint_path)
            n_kw = {}
            for key, value in kw.items():
                if key.startswith('bert.'):
                    key = key[5:]
                n_kw[key] = value
            self.bert_model.load_state_dict(n_kw, strict=False)
        self.device = device
        self.batch_size = batch_size
        self.bert_model.eval()

    @torch.no_grad()
    def encode_dialogue(self, dialogue: Dialogue) -> np.array:
        utterances = [utt.utterance for utt in dialogue]
        tokenized = self.bert_tokenizer(utterances, padding=True,
                                        return_tensors='pt')
        tokenized = {key: value.to(self.device) for key, value in
                     tokenized.items()}
        embeddings = self.bert_model(**tokenized).pooler_output.cpu().numpy()
        return embeddings

    @torch.no_grad()
    def encode_dataset(self, dialogues: DialogueDataset) -> np.array:
        all_embeddings = []
        for i in trange(0, len(dialogues.utterances), self.batch_size):
            tokenized = self.bert_tokenizer(
                dialogues.utterances[i:i + self.batch_size],
                padding=True, return_tensors='pt')
            tokenized = {key: value.to(self.device) for key, value in
                         tokenized.items()}
            embeddings = self.bert_model(
                **tokenized).pooler_output.cpu().numpy()
            all_embeddings.append(embeddings)
        return np.concatenate(all_embeddings, axis=0)

    def encode_new_dialogue(self, dialogue: Dialogue):
        return self.encode_dialogue(dialogue)

    def encode_new_dataset(self, dialogues: DialogueDataset):
        return self.encode_dataset(dialogues)