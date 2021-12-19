import pickle
import timeit
from typing import Tuple, Dict

import numpy as np

import torch
from torch import optim
import torch.nn as nn

import transformers
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, BartForConditionalGeneration, PretrainedConfig
from transformers import BartTokenizerFast
from transformers.models.bart.configuration_bart import BartConfig

from model import BartSummaryModelV2
from utils import collate_fn, freeze, unfreeze_all, PrintInfo
from dataset import SummaryDataset
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import wandb

backward_steps = 64
eval_steps = 10000

def train_step(model, batch) -> Tuple[torch.FloatTensor, Dict[str, float]]:
    alpha = 0.5

    input_ids = batch["input_ids"].cuda()  # (B, L_src)
    attention_mask = batch["attention_mask"].cuda()  # (B, L_src)
    answers = batch["answers"].cuda() # 추출요약 (B, 3)
    labels = batch["labels"].cuda()   # 생성요약 (B, L_tgt)

    ext_out = model.classify(input_ids=input_ids, attention_mask=attention_mask, labels=answers)
    gen_out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    total_loss = alpha * ext_out.loss + (1-alpha) * gen_out.loss

    return total_loss, {"ext_loss": ext_out.loss.item(), "gen_loss": gen_out.loss.item()}

def train_loop(model, train_dl, optimizer, epoch: int = -1, prev_step: int = 0) -> int:
    step = prev_step
    model.train()
    optimizer.zero_grad()
    ext_losses = []
    gen_losses = []

    for batch in tqdm(train_dl):
        loss, returned_dict = train_step(model, batch)
        ext_losses.append(returned_dict["ext_loss"])
        gen_losses.append(returned_dict["gen_loss"])
        loss.backward()
        step += 1

        if (step+1) % backward_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            train_metrics = {"train/ext_loss": np.mean(ext_losses), "train/gen_loss": np.mean(gen_losses), "step": step}
            wandb.log(train_metrics)

        if (step+1) % eval_steps == 0:
            eval_metrics = eval_loop()
            eval_metrics = {("eval/" + k): v for k, v in eval_metrics.items()}
            eval_metrics["step"] = step
            wandb.log(eval_metrics)

    return step

def eval_loop(model, eval_dl) -> Dict[str, float]:
    model.eval()

    ext_loss = 0.0
    gen_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch in tqdm(eval_dl):
            size = len(input_ids)

            input_ids = batch["input_ids"].cuda()  # (B, L_src)
            attention_mask = batch["attention_mask"].cuda()  # (B, L_src)
            answers = batch["answers"].cuda() if "answers" in batch.keys() else None # 추출요약 (B, 3)
            labels = batch["labels"].cuda() if "labels" in batch.keys() else None

            ext_out = model.classify(input_ids=input_ids, attention_mask=attention_mask, labels=answers)
            gen_out = model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # weighted sum
            ext_loss = (n * ext_loss + size * ext_out.loss.item()) / (n + size)
            gen_loss = (n * gen_loss + size * gen_out.loss.item()) / (n + size)
            n += size

    return {
        "ext_loss": ext_loss,
        "gen_loss": gen_loss,
    }


def main():

    wandb.init(project="easybart")

    # load config, tokenizer, model
    MODEL_NAME = "gogamza/kobart-summarization"
    config = BartConfig.from_pretrained(MODEL_NAME)
    tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
    model = BartSummaryModelV2.from_pretrained(MODEL_NAME)

    # load dataset, dataloader
    train_path = "/opt/ml/dataset/Training/train.parquet"
    eval_path = "/opt/ml/dataset/Training/train.parquet"

    train_dataset = SummaryDataset(train_path, tokenizer, is_train=True)
    eval_dataset = SummaryDataset(eval_path, tokenizer, is_train=True)

    print(f"train_dataset length: {len(train_dataset)}, eval_dataset length: {len(eval_dataset)}")

    TRAIN_BATCH_SIZE = 4
    EVAL_BATCH_SIZE = 8

    train_dataloader = DataLoader(train_dataset, 
        TRAIN_BATCH_SIZE, 
        shuffle=True, 
        collate_fn=lambda x: collate_fn(x, pad_token_idx=tokenizer.pad_token_id)
    )
    eval_dataloader = DataLoader(eval_dataset, 
        EVAL_BATCH_SIZE, 
        shuffle=False, 
        collate_fn=lambda x: collate_fn(x, pad_token_idx=tokenizer.pad_token_id)
    )

    # optimizer
    # TODO: LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

    # train loop
    model.cuda()
    model.train()

    EPOCH = 16
    ext_losses = []
    gen_losses = []

    total_steps = 0
    optimizer.zero_grad()

    for epoch in range(EPOCH):
        print("=" * 10 + "Epoch " + str(epoch) + " has started!" + "=" * 10)
        total_steps = train_loop(model, train_dataloader, optimizer, total_steps)

        # epoch이 끝나면 누적 loss 데이터 전체 저장
        with open(f"./ext_losses_ep_{epoch}.pkl", "wb") as f:
            pickle.dump(ext_losses, f)
        with open(f"./gen_losses_ep_{epoch}.pkl", "wb") as f:
            pickle.dump(gen_losses, f)

    model.save_pretrained("./saved")


if __name__ == '__main__':
    main()