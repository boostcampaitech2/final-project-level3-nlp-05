import os
import json
import argparse
from datetime import date, timedelta

import pandas as pd

import torch
from torch.utils.data import DataLoader

from transformers import BartTokenizerFast, BartConfig

from model import BartSummaryModelV2
from dataset import TestDataset
from utils import collate_fn

from tqdm import tqdm

categories = {
    'society': '사회',
    'politics': '정치',
    'economic': '경제',
    'foreign': '국제',
    'culture': '문화',
    'entertain': '연예',
    'sports': '스포츠',
    'digital': 'IT'
}

def extract_sentences(
    input_ids: torch.FloatTensor,
    eos_positions: torch.LongTensor,
    ext_ids: torch.LongTensor,
    tokenizer: BartTokenizerFast,  
):
    PAD = tokenizer.pad_token_id
    gen_batch_inputs = []
    attention_mask = []

    for i in range(input_ids.size(0)):
        ids = ext_ids[i][ext_ids[i] >= 0].tolist()
        sentences = [torch.tensor([tokenizer.bos_token_id])]
        for idx in ids:
            from_pos = 1 if idx == 0 else (eos_positions[i, idx-1].item() + 1)
            to_pos = (eos_positions[i, idx].item() + 1)
            
            ext_sentence = input_ids[i, from_pos:to_pos].clone().detach()
            sentences.append(ext_sentence)
        sentences = torch.cat(sentences, dim=0)
        gen_batch_inputs.append(sentences)
        attention_mask.append(torch.ones(len(sentences)))

    gen_batch_inputs = torch.nn.utils.rnn.pad_sequence(gen_batch_inputs, padding_value=PAD, batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=PAD, batch_first=True)
    return {
        "input_ids": gen_batch_inputs,
        "attention_mask": attention_mask,
    }

def inference(args):
    # device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # tokenizer, model
    tokenizer = BartTokenizerFast.from_pretrained(args.tokenizer)
    model = BartSummaryModelV2.from_pretrained(args.model_dir)  # 일단
    
    # get data
    data_dir = f"./data/{args.date}"
    file_name = f"clustering_{args.date}_{categories[args.category]}.json"
    save_file_name = f"summary_{args.date}_{categories[args.category]}.json"
    if os.path.isfile(os.path.join(data_dir, save_file_name)):
        print(f'{save_file_name} is already generated.')
        return
        
    test_file = os.path.join(data_dir, file_name)
    test_dataset = TestDataset(test_file, tokenizer)
    
    print("test_dataset length:", len(test_dataset))
    
    BATCH_SIZE = 8
    test_dataloader = DataLoader(test_dataset, 
        BATCH_SIZE, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, pad_token_idx=tokenizer.pad_token_id, sort_by_length=False),
        drop_last=False
    )

    model.to(device)
    model.eval()
    
    final_sents = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(device)  # (B, L_src)
            attention_mask = batch["attention_mask"].to(device)  # (B, L_src)
            eos_positions = batch["eos_positions"].to(device)

            ext_out = model.classify(input_ids=input_ids, attention_mask=attention_mask)

            # 일단 무조건 3개 이상 나오고, top 3개만 자른다고 가정
            TOPK = 3
            top_ext_ids = torch.argsort(ext_out.logits, dim=-1, descending=True)[:, :TOPK]  # (B, TOPK)

            gen_batch = extract_sentences(input_ids, eos_positions, top_ext_ids, tokenizer)

            summary_ids = model.generate(
                input_ids=gen_batch["input_ids"].to(device), 
                attention_mask=gen_batch["attention_mask"].to(device), 
                num_beams=8, 
                max_length=128, 
                min_length=4,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )  # args로 받기
            summary_sent = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
            final_sents.extend(summary_sent)
            
    print("Inference completed!")
    test_id = test_dataset.get_id_column()
    
    assert len(test_id) == len(final_sents)
    
    test_title = test_dataset.get_title_column()
    test_category = test_dataset.get_category_column()

    output = []
    for i, id in enumerate(test_id):
        output.append({
            "id": id,
            "title": test_title[i],
            "category": test_category[i],
            "summary": final_sents[i]
        })

    # output.to_json('./summary.json')  # json 으로 저장
    with open(os.path.join(data_dir, save_file_name), 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir', type=str, default="./saved")
    parser.add_argument('--tokenizer', type=str, default="gogamza/kobart-summarization")
    parser.add_argument('--data_dir', type=str, default="/opt/ml/dataset/Test")
    parser.add_argument('--date', type=str, default=(date.today() - timedelta(1)).strftime("%Y%m%d")) # 어제날짜
    parser.add_argument('--category', type=str, default='society', choices=["society", "politics", "economic", "foreign", "culture", "entertain", "sports", "digital"])
    args = parser.parse_args()
    
    inference(args)
