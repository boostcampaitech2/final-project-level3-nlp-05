import json

import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import pandas as pd
import os

from transformers import PreTrainedTokenizerFast, BartTokenizerFast

class SummaryDataset(Dataset):
    def __init__(
        self, 
        file_path: str, 
        tokenizer: BartTokenizerFast, 
        max_seq_len: int = 1024, 
        is_train: bool = False,
    ):
        self.path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.is_train = is_train

        self._file_ext = os.path.splitext(self.path)[-1].lower()
        if self._file_ext == ".parquet":
            self.raw_data = pq.read_table(self.path)
        elif self._file_ext == ".json":
            self.raw_data = pd.read_json(self.path)
            self.raw_data = self._reorganize_text(self.raw_data)
        else:
            raise ValueError("File extension must be .parquet or .json")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        if self._file_ext == ".parquet":
            title = self.raw_data["title"][idx].as_py()
            input_sentences = self.raw_data["text"][idx].as_py()
        else:
            title = self.raw_data["title"][idx]
            input_sentences = self.raw_data["text"][idx]

        if self.is_train:
            if self._file_ext == ".parquet":
                target_sentence = self.raw_data["abstractive"][idx][0].as_py()
                target_ids = self.raw_data["extractive"][idx].as_py()
            else:
                target_sentence = self.raw_data["abstractive"][idx][0]
                target_ids = self.raw_data["extractive"][idx]
        else:
            target_sentence = None
            target_ids = None

        input_ids = []
        eos_positions = []

        # TODO: 제목 추가하기 + Special Token 처리 방법 고민...
        if title is not None and title != "":
            pass
        
        input_ids.append(self.tokenizer.bos_token_id)

        # <s> $sentence_A </s> $sentence_B </s> sentence_C </s> ...
        for sentence in input_sentences:
            input_ids.extend(self.tokenizer.encode(sentence))
            eos_positions.append(len(input_ids))
            input_ids.append(self.tokenizer.eos_token_id)

        if target_ids is not None:
            target_ids = [idx for idx in target_ids if idx is not None]
        if self.is_train and len(target_ids) == 0:
            # safety checks!
            target_ids = [-1]

        # truncation
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len-1] + [self.tokenizer.eos_token_id]
            attention_mask = [1.] * self.max_seq_len
            num_eos = input_ids.count(self.tokenizer.eos_token_id)
            if target_ids is not None:
                target_ids = [id for id in target_ids if id < num_eos]
        else:
            attention_mask = [1.] * len(input_ids)

        if target_sentence is not None:
            labels = [self.tokenizer.bos_token_id] + self.tokenizer.encode(target_sentence) + [self.tokenizer.eos_token_id]
        else:
            labels = None

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
            "eos_positions": torch.tensor(eos_positions, dtype=torch.long),
            "answers": torch.tensor(target_ids, dtype=torch.long) if target_ids else None, # exists if not is_train
            "labels": torch.tensor(labels, dtype=torch.long) if labels else None, # exists if not is_train
        }

    def get_df(self):
        return self.raw_data.to_pandas()

    def get_id_column(self):
        return self.raw_data['id'].tolist()

    def get_category_column(self):
        return self.raw_data['category'].tolist()

    def get_title_column(self):
        return self.raw_data['title'].tolist()

    def _to_list_str(self, paragraphs):
        result = []
        for paragraph in paragraphs:
            if len(paragraph) < 1: 
                # no sentence in paragraph
                continue
            result.extend([sentence["sentence"] for sentence in paragraph])
        return result

    def _reorganize_text(self, raw_data):
        raw_data.loc[:, "text"] = raw_data.text.apply(self._to_list_str)
        return raw_data
    
    
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

    def get_category_column(self):
        return self.raw_data['category'].tolist()

    def get_title_column(self):
        return self.raw_data['title'].tolist()
    
