import argparse
from dataset import TestDataset
from transformers import BartTokenizerFast, BartConfig
from torch.utils.data import DataLoader
from utils import collate_fn
import os
from model import BartSummaryModelV2
import torch
from tqdm import tqdm
import pandas as pd


def inference(args):
    # tokenizer, model
    tokenizer = BartTokenizerFast.from_pretrained(args.tokenizer)
    model = BartSummaryModelV2.from_pretrained(args.model_dir)  # 일단
    
    # get data
    test_file = os.path.join(args.data_dir, "test.json")
    test_dataset = TestDataset(test_file, tokenizer)
    
    print("test_dataset length:", len(test_dataset))
    
    BATCH_SIZE = 8
    test_dataloader = DataLoader(test_dataset, 
        BATCH_SIZE, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, pad_token_idx=tokenizer.pad_token_id, sort_by_length=False),
        drop_last=False
    )
    # pad, attention 맞는지 확인하기!

    model.cuda()
    model.eval()
    
    final_sents = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].cuda()  # (B, L_src)
            attention_mask = batch["attention_mask"].cuda()  # (B, L_src)

            # 일단 ext 안 거치고 바로 generate
            summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=8, max_length=128, min_length=4)  # args로 받기
            summary_sent = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
            final_sents.extend(summary_sent)
            
    print("Inference completed!")
    test_id = test_dataset.get_id_column()
    
    assert len(test_id) == len(final_sents)
    
    output = pd.DataFrame({'id': test_id,'summary': final_sents})
    output.to_json('./summary.json')  # json 으로 저장
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 나중에 수정하기
    parser.add_argument('--model_dir', type=str, default="./saved")
    parser.add_argument('--tokenizer', type=str, default="gogamza/kobart-summarization")
    parser.add_argument('--data_dir', type=str, default="/opt/ml/dataset/Test")
    args = parser.parse_args()
    
    inference(args)