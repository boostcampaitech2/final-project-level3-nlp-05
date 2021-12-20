import os
import json
import argparse
from datetime import date, timedelta
from typing import Optional, List

import pandas as pd

import torch
from torch.utils.data import DataLoader

from transformers import BartTokenizerFast, BartConfig

from model import BartSummaryModelV2
from dataset import SummaryDataset, TestDataset
from utils import collate_fn, str2bool

from tqdm import tqdm
import glob


def get_top_k_sentences(logits: torch.FloatTensor, eos_positions: torch.LongTensor, k: int = 3):
    returned_tensor = []
    top_ext_ids = torch.argsort(logits, dim=-1, descending=True)
    num_sentences = torch.sum(torch.gt(eos_positions, 0), dim=-1, dtype=torch.long)

    for i in range(len(top_ext_ids)):
        top_ext_id = top_ext_ids[i]
        top_ext_id = top_ext_id[top_ext_id < num_sentences[i]]
        top_ext_id = top_ext_id[:k]
        top_k, _ = torch.sort(top_ext_id)
        returned_tensor.append(top_k.unsqueeze(0))
    
    returned_tensor = torch.cat(returned_tensor, dim=0)

    return returned_tensor


def concat_json(data_dir, date):
    '''combine files for each category into one whole json file'''
    dir_path = os.path.join(data_dir, date)

    save_file_name = f"{dir_path}/cluster_for_summary_{date}.json"
    if os.path.isfile(save_file_name):
        print(f'{save_file_name} has been already generated.')
        return

    result = []
    for file in glob.glob(f"{dir_path}/cluster_for_summary*.json"):
        with open(file, "r") as f:
            result.extend(json.load(f))
    with open(save_file_name, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print("Concatenation Completed!")

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
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, padding_value=0, batch_first=True)

    return {
        "input_ids": gen_batch_inputs,
        "attention_mask": attention_mask,
    }


def predict(args, model, test_dl, tokenizer) -> List[str]:

    device = torch.device("cpu") if args.no_cuda or not torch.cuda.is_available() else torch.device("cuda")

    model.to(device)
    model.eval()
    
    pred_sentences = []
    pred_ext_ids = []

    with torch.no_grad():
        for batch in tqdm(test_dl):
            input_ids = batch["input_ids"].clone().to(device)  # (B, L_src)
            attention_mask = batch["attention_mask"].clone().to(device)  # (B, L_src)
            # eos_positions = batch["eos_positions"].clone().to(device)

            ext_out = model.classify(input_ids=input_ids, attention_mask=attention_mask)

            # TODO: use different k values
            # TODO: implement different criteria (such as probability)!
            top_ext_ids = get_top_k_sentences(
                logits=ext_out.logits.clone().detach().cpu(), 
                eos_positions=batch["eos_positions"], 
                k = args.num_extraction,
            )
            pred_ext_ids.extend(top_ext_ids.tolist())
            
            gen_batch = extract_sentences(batch["input_ids"], batch["eos_positions"], top_ext_ids, tokenizer)

            summary_ids = model.generate(
                input_ids=gen_batch["input_ids"].to(device), 
                attention_mask=gen_batch["attention_mask"].to(device), 
                num_beams=args.num_beams, 
                max_length=args.max_length, 
                min_length=args.min_length,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            
            summary_sent = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
            pred_sentences.extend(summary_sent)

    return pred_sentences, pred_ext_ids


def main(args):
    # tokenizer, model
    tokenizer = BartTokenizerFast.from_pretrained(args.tokenizer)
    model = BartSummaryModelV2.from_pretrained(args.model_dir)
    
    # get data
    data_dir = os.path.join(args.data_dir, args.date)

    save_file_name = f"summary_{args.date}.json"
    if os.path.isfile(os.path.join(data_dir, save_file_name)):
        print(f'{save_file_name} has been already generated.')
        return
        
    file_name = f"cluster_for_summary_{args.date}.json"
    test_file = os.path.join(data_dir, file_name)

    test_dataset = SummaryDataset(test_file, tokenizer)
    
    print(f"test dataset length: {len(test_dataset)}")
    
    test_dl = DataLoader(
        test_dataset, 
        args.per_device_eval_batch_size, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, pad_token_idx=tokenizer.pad_token_id, sort_by_length=False),
        drop_last=False,
    )

    pred_sents, pred_ext_ids = predict(args, model, test_dl, tokenizer)
            
    print("Inference completed!")
    test_id = test_dataset.get_id_column()
    
    assert len(test_id) == len(pred_sents), "length of test_id and pred_sents do not match"
    
    test_title = test_dataset.get_title_column()
    test_category = test_dataset.get_category_column()

    output = []
    for i, id in enumerate(test_id):
        output.append({
            "id": id,
            "title": test_title[i],
            "category": test_category[i],
            "extract_ids": pred_ext_ids[i],
            "summary": pred_sents[i]
        })

    # output.to_json('./summary.json')  # json 으로 저장
    with open(os.path.join(data_dir, save_file_name), 'w', encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir', type=str, default="./saved")
    parser.add_argument('--tokenizer', type=str, default="gogamza/kobart-summarization")
    parser.add_argument('--data_dir', type=str, default="/opt/ml/dataset/Test")
    parser.add_argument('--date', type=str, default=(date.today() - timedelta(1)).strftime("%Y%m%d")) # 어제날짜

    parser.add_argument("--per_device_eval_batch_size", default=8, type=int, help="inference batch size per device (default: 8)")

    parser.add_argument('--num_beams', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--min_length', type=Optional[int])
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=Optional[int])

    parser.add_argument("--no_cuda", default=False, type=str2bool, help="run on cpu if True")

    parser.add_argument("--num_extraction", type=int, default = 3)

    args = parser.parse_args()
    
    concat_json(args.data_dir, args.date)
    main(args)
