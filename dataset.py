import json

import torch
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizerFast

class SummaryDataset(Dataset):
    def __init__(self, path: str, tokenizer: PreTrainedTokenizerFast, max_seq_len: int = 1024):
        self.path = path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        with open(path, "r") as json_files:
            json_list = list(json_files)

        raw_data = []
        for json_str in json_list:
            result = json.loads(json_str)
            raw_data.append(result)

        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        input_sentences = self.raw_data[idx]["article_original"]
        target_sentence = self.raw_data[idx]["abstractive"]
        target_ids = self.raw_data[idx]["extractive"]

        input_ids = []
        bos_positions = []

        for sentence in input_sentences:
            bos_positions.append(len(input_ids))
            input_ids.append(self.tokenizer.bos_token_id)
            input_ids.extend(self.tokenizer.encode(sentence))
        input_ids.append(self.tokenizer.eos_token_id)

        # TODO: trunctation
        attention_mask = None
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:-1] + [self.tokenizer.eos_token_id]
            attention_mask = [1] * self.max_seq_len
        else:
            attention_mask = [1] * len(input_ids)

        answer_positions = [bos_positions[i] for i in target_ids if bos_positions[i] < self.max_seq_len]
        labels = [self.tokenizer.bos_token_id] + self.tokenizer.encode(target_sentence) + [self.tokenizer.eos_token_id]

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
            "bos_positions": torch.tensor(bos_positions),
            "answer_positions": torch.tensor(answer_positions),
        }

    def get_df(self):
        import pandas as pd
        return pd.DataFrame(self.raw_data)