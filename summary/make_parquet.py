import json
import argparse
from typing import List

import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

from utils import combine_sentences

def extract_train_set(
    from_path: str = "train_original.json", 
    to_path: str = "train.json",
):

    with open(from_path, "r") as f:  # ai hub 신문기사 json 파일
        data = json.load(f)
        data = data["documents"]

    with open(to_path, "w") as f:  # 새로 저장
        json.dump(data, f, indent=4, ensure_ascii=False)  # 한글이라 ensure_ascii=False

def to_parquet(df: pd.DataFrame, save_path: str="train.parquet"):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, save_path)  # preserve_index=False (omit index)
    return table

def main(args):
    extract_train_set(args.original_path, args.json_save_path)
    news_df = pd.read_json(args.json_save_path)
    news_df = news_df[['id', 'char_count', 'title', 'text', 'extractive', 'abstractive']]
    news_df["text"] = news_df["text"].apply(combine_sentences)
    to_parquet(news_df, args.parquet_save_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Create dataset for train/eval/test")

    parser.add_argument("--original_path", type=str, default="/opt/ml/dataset/Training/train_original.json")
    parser.add_argument("--json_save_path", type=str, default="/opt/ml/dataset/Training/train.json")
    parser.add_argument("--parquet_save_path", type=str, default="/opt/ml/dataset/Training/train.parquet")

    args = parser.parse_args()

    main(args)
