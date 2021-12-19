import os
import json
from typing import Optional, Tuple, Dict, NoReturn
import argparse

import numpy as np

import torch
from torch.cuda import is_available
import torch.nn as nn
import torch.optim as optim

import transformers
from transformers import BartTokenizerFast
from transformers.models.bart.configuration_bart import BartConfig

from model import BartSummaryModelV2
from inference import predict
from utils import set_all_seeds, collate_fn, freeze, unfreeze_all, PrintInfo, str2bool, np_sigmoid
from dataset import SummaryDataset
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

alpha = 0.5

def train_step(model, batch, device) -> Tuple[torch.FloatTensor, Dict[str, float]]:

    input_ids = batch["input_ids"].to(device)  # (B, L_src)
    attention_mask = batch["attention_mask"].to(device)  # (B, L_src)
    answers = batch["answers"].to(device) # 추출요약 (B, 3)
    labels = batch["labels"].to(device) # 생성요약 (B, L_tgt)

    ext_out = model.classify(input_ids=input_ids, attention_mask=attention_mask, labels=answers)
    gen_out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    total_loss = alpha * ext_out.loss + (1-alpha) * gen_out.loss

    return total_loss, {"ext_loss": ext_out.loss.item(), "gen_loss": gen_out.loss.item(), "ext_logits": ext_out.logits}

def train_loop(args, model, train_dl, eval_dl, optimizer, prev_step: int = 0) -> int:
    
    step = prev_step

    model.train()
    optimizer.zero_grad()
    ext_losses = []
    gen_losses = []
    all_logits = []

    if args.use_wandb:
        import wandb

    if args.do_train:

        for batch in tqdm(train_dl):

            model.train()
            device = torch.device("cpu") if args.no_cuda or not torch.cuda.is_available() else torch.device("cuda")

            loss, returned_dict = train_step(model, batch, device)
            loss.backward()
            ext_losses.append(returned_dict["ext_loss"])
            gen_losses.append(returned_dict["gen_loss"])
            all_logits.append(returned_dict["ext_logits"].detach().cpu().numpy())
            step += 1

            if (step+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                all_logits = np.hstack(all_logits)
                all_probs = np_sigmoid(all_logits)
                hist = np.histogram(all_probs)

                train_metrics = {
                    "train/ext_loss": np.mean(ext_losses), 
                    "train/gen_loss": np.mean(gen_losses), 
                    "train/probs": wandb.Histogram(np_histogram=hist),
                    "step": step,
                }
                if args.use_wandb:
                    wandb.log(train_metrics)

                ext_losses = []
                gen_losses = []
                all_logits = []

            if args.do_eval and (step+1) % args.eval_steps == 0:
                eval(args, model, eval_dl, step)

    return step


def eval(args, model, eval_dl, step) -> Dict[str, float]:
    device = torch.device("cpu") if args.no_cuda or not torch.cuda.is_available() else torch.device("cuda")
    eval_metrics = eval_loop(model, eval_dl, device)
    eval_metrics = {("eval/" + k): v for k, v in eval_metrics.items()}
    eval_metrics["step"] = step

    print(eval_metrics)
    if args.use_wandb:
        import wandb
        wandb.log(eval_metrics)
    
    return eval_metrics


def eval_loop(model, eval_dl, device) -> Dict[str, float]:

    if args.use_wandb:
        import wandb

    model.eval()

    ext_loss = 0.0
    gen_loss = 0.0
    all_logits = []
    n = 0

    with torch.no_grad():
        for batch in tqdm(eval_dl):
            input_ids = batch["input_ids"].to(device)  # (B, L_src)
            attention_mask = batch["attention_mask"].to(device)  # (B, L_src)
            answers = batch["answers"].to(device) if "answers" in batch.keys() else None # 추출요약 (B, 3)
            labels = batch["labels"].to(device) if "labels" in batch.keys() else None

            ext_out = model.classify(input_ids=input_ids, attention_mask=attention_mask, labels=answers)
            gen_out = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            all_logits.append(ext_out.logits.cpu().numpy())

            # weighted sum
            size = len(input_ids)
            ext_loss = (n * ext_loss + size * ext_out.loss.item()) / (n + size)
            gen_loss = (n * gen_loss + size * gen_out.loss.item()) / (n + size)
            n += size

    all_logits = np.hstack(all_logits)
    all_probs = np_sigmoid(all_logits)
    hist = np.histogram(all_probs)

    return {
        "ext_loss": ext_loss,
        "gen_loss": gen_loss,
        "probs": wandb.Histogram(np_histogram=hist) if args.use_wandb else None,
    }


def main(args):

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
        )

    if args.seed:
        set_all_seeds(args.seed, verbose=True)

    # load config, tokenizer, model
    MODEL_NAME = "gogamza/kobart-summarization"
    config = BartConfig.from_pretrained(MODEL_NAME)
    tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
    model = BartSummaryModelV2.from_pretrained(MODEL_NAME)

    # load dataset, dataloader
    train_path = "/opt/ml/dataset/Training/train.parquet"
    eval_path  = "/opt/ml/dataset/Validation/valid.parquet"

    train_dataset = SummaryDataset(train_path, tokenizer, is_train=True) if args.do_train else None
    eval_dataset  = SummaryDataset(eval_path, tokenizer, is_train=True) if args.do_eval else None

    if train_dataset is not None:
        print(f"train_dataset length: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"eval_dataset length: {len(eval_dataset)}")

    train_dl = DataLoader(
        train_dataset, 
        args.per_device_train_batch_size, 
        shuffle=True, 
        collate_fn=lambda x: collate_fn(x, pad_token_idx=tokenizer.pad_token_id),
    ) if args.do_train else None

    eval_dl = DataLoader(
        train_dataset if eval_dataset is None else eval_dataset, 
        args.per_device_eval_batch_size, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, pad_token_idx=tokenizer.pad_token_id),
    ) if args.do_eval else None

    # optimizer
    # TODO: LR scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        betas=[args.adam_beta1, args.adam_beta2],
    )

    # train loop
    if not args.no_cuda:
        device = torch.device("cpu") if args.no_cuda or not torch.cuda.is_available() else torch.device("cuda")
        model.to(device)
    model.train()

    total_steps = 0
    optimizer.zero_grad()

    if args.do_train:
        for epoch in range(int(args.num_train_epochs)):
            print("=" * 10 + "Epoch " + str(epoch+1) + " has started! " + "=" * 10)
            total_steps = train_loop(args, model, train_dl, eval_dl, optimizer, total_steps)
            
            if args.do_predict:
                print("=" * 10 + "Epoch " + str(epoch+1) + " predict has started! " + "=" * 10)
                pred, _ = predict(args, model, eval_dl, tokenizer)
                with open(os.path.join(args.output_dir, f"pred_epoch_{epoch}.json"), 'w', encoding="utf-8") as f:
                    json.dump(pred, f, ensure_ascii=False)

            # save the trained model at the end of every epoch
            model.save_pretrained(args.output_dir)
    
    if args.do_eval:
        print("=" * 10 + "The final evaluation loop has started!" + "=" * 10)
        eval(args, model, eval_dl, total_steps)

    if args.do_predict:
        print("=" * 10 + "The final prediction loop has started!" + "=" * 10)
        pred_sents, _ = predict(args, model, eval_dl, tokenizer)
        with open(os.path.join(args.output_dir, f"pred_final.json"), 'w', encoding="utf-8") as f:
            json.dump(pred_sents, f, ensure_ascii=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train model.")

    parser.add_argument("--do_train", default=True, type=str2bool, help="run train loop if True")
    parser.add_argument("--do_eval", default=True, type=str2bool, help="run evaluation loop if True")
    parser.add_argument("--do_predict", default=True, type=str2bool, help="run predict loop if True")

    parser.add_argument("--output_dir", default="./saved", type=str, help="path to save the trained model")
    
    parser.add_argument("--per_device_train_batch_size", default=4, type=int, help="train batch size per device (default: 4)")
    parser.add_argument("--per_device_eval_batch_size", default=8, type=int, help="eval batch size per device (default: 8)")
    parser.add_argument("--num_train_epochs", default=1.0, type=float, help="num train epochs")
    parser.add_argument("--eval_steps", default=500, type=int, help="num train epochs")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="num gradient accumulation steps")

    parser.add_argument("--no_cuda", default=False, type=str2bool, help="run on cpu if True")
    parser.add_argument("--seed", type=int, help="random seed number")

    parser.add_argument("--learning_rate", default=1e-05, type=float, help="learning rate")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weigth decay in AdamW optimizer")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="beta1 in AdamW optimizer")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="beta2 in AdamW optimizer")

    # for predict loop
    parser.add_argument('--num_beams', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--min_length', type=Optional[int])
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=Optional[int])

    parser.add_argument("--use_wandb", default=True, type=str2bool, help="use wandb if True")
    parser.add_argument("--wandb_run_name", default="run", type=str, help="wandb run name")
    parser.add_argument("--wandb_project", default="easybart", type=str, help="wandb project name")
    parser.add_argument("--wandb_entity", default="this-is-real", type=str, help="wandb entity name")
    
    args = parser.parse_args()

    main(args)