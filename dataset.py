import json

import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import pandas as pd

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

        input_ids = [self.tokenizer.bos_token_id]

        for sentence in input_sentences:
            input_ids.extend(self.tokenizer.encode(sentence))
            input_ids.append(self.tokenizer.eos_token_id)

        attention_mask = None

        if target_ids is not None:
            for i in range(len(target_ids)):
                if not isinstance(target_ids[i], int):  # such as None
                    target_ids[i] = -1  # same as PAD in collate_fn

        target_ids = torch.tensor(target_ids, dtype=torch.long)
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len-1] + [self.tokenizer.eos_token_id]
            attention_mask = [1.] * self.max_seq_len
            num_eos = input_ids.count(self.tokenizer.eos_token_id)
            target_ids = target_ids[target_ids < num_eos]
        else:
            attention_mask = [1.] * len(input_ids)

        labels = [self.tokenizer.bos_token_id] + self.tokenizer.encode(target_sentence) + [self.tokenizer.eos_token_id]

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "answers": target_ids,
            "labels": torch.tensor(labels)
        }

    def get_df(self):
        return self.raw_data.to_pandas()
    
    
class TestDataset(Dataset):
    def __init__(self, json_path: str, tokenizer: BartTokenizerFast, max_seq_len: int = 1024):
        self.path = json_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.raw_data = pd.read_json(json_path)
        self.raw_data = self.raw_data[:100]  # sample

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        input_sentence = self.raw_data["text"][idx]

        # input_ids: bos t t t eos ?
        input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(" ".join(input_sentence)) + [self.tokenizer.eos_token_id]
        attention_mask = [1.] * len(input_ids)
        
        # truncate
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len-1] + [self.tokenizer.eos_token_id]
            attention_mask = [1.] * self.max_seq_len        

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }

    def get_df(self):
        return self.raw_data
    
    def get_id_column(self):
        return self.raw_data['id'].tolist()
    