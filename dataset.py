import json

import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq

from transformers import PreTrainedTokenizerFast, BartTokenizerFast

class SummaryDataset(Dataset):
    def __init__(self, parquet_path: str, tokenizer: BartTokenizerFast, max_seq_len: int = 1024):
        self.path = parquet_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.raw_data = pq.read_table(parquet_path)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        input_sentences = self.raw_data["text"][idx].as_py()
        target_sentence = self.raw_data["abstractive"][idx][0].as_py()
        target_ids = self.raw_data["extractive"][idx].as_py()

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
        return self.raw_data.to_pandas()