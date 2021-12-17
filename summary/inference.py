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
from datetime import date, timedelta
import json

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

def _make_generation_input(batch_input_ids, batch_eos_positions, batch_ext_ids, tokenizer):
    '''make input data for generation'''
    gen_input_ids = []
    gen_attention_mask = []
    for i in range(len(batch_input_ids)):
        input_ids = [tokenizer.bos_token_id]
        eos_positions = torch.cat((torch.tensor([0]).cuda(), batch_eos_positions[i]))

        for id in batch_ext_ids[i]:
            # sentence tokens
            sent_ids = batch_input_ids[i][eos_positions[id]+1:eos_positions[id+1]]
            input_ids.extend(sent_ids)
        input_ids.append(tokenizer.eos_token_id)
        attention_mask = [1.] * len(input_ids)
        print(len(input_ids), len(attention_mask))
        
        gen_input_ids.append(input_ids)
        gen_attention_mask.append(attention_mask)

    return torch.tensor(gen_input_ids), torch.tensor(gen_attention_mask)

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
    # test_file = os.path.join(args.data_dir, "test.json")
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
    # pad, attention 맞는지 확인하기!

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

            gen_input_ids, gen_attention_mask = _make_generation_input(input_ids, eos_positions, top_ext_ids, tokenizer)

            # 일단 ext 안 거치고 바로 generate
            summary_ids = model.generate(
                input_ids=gen_input_ids, 
                attention_mask=gen_attention_mask, 
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
    #output = pd.DataFrame({'id': test_id,'summary': final_sents})
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
    
    # 나중에 수정하기
    parser.add_argument('--model_dir', type=str, default="./saved")
    parser.add_argument('--tokenizer', type=str, default="gogamza/kobart-summarization")
    parser.add_argument('--data_dir', type=str, default="/opt/ml/dataset/Test")
    parser.add_argument('--date', type=str, default=(date.today() - timedelta(1)).strftime("%Y%m%d")) # 어제날짜
    parser.add_argument('--category', type=str, default='society', choices=["society", "politics", "economic", "foreign", "culture", "entertain", "sports", "digital"])
    args = parser.parse_args()
    
    inference(args)
