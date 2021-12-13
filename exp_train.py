import numpy as np
import timeit

import torch
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
import pickle


# load config, tokenizer, model
MODEL_NAME = "gogamza/kobart-summarization"
config = BartConfig.from_pretrained(MODEL_NAME)
tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
model = BartSummaryModelV2.from_pretrained(MODEL_NAME)

# oad dataset, dataloader
train_path = "/opt/ml/dataset/Training/train.parquet"
train_dataset = SummaryDataset(train_path, tokenizer)

BATCH_SIZE = 4
train_dataloader = DataLoader(train_dataset, 
    BATCH_SIZE, 
    shuffle=True, 
    collate_fn=lambda x: collate_fn(x, pad_token_idx=tokenizer.pad_token_id)
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

model.cuda()
model.train()

EPOCH = 16
ext_losses = []
gen_losses = []

backward_steps = 64
optimizer.zero_grad()
for epoch in range(EPOCH):
    i = 0
    for idx, batch in tqdm(enumerate(train_dataloader)):
        input_ids = batch["input_ids"].cuda()  # (B, L_src)
        attention_mask = batch["attention_mask"].cuda()  # (B, L_src)
        answers = batch["answers"].cuda() # 추출요약 (B, 3)
        labels = batch["labels"].cuda()   # 생성요약 (B, L_tgt)

        ext_out = model.classify(input_ids=input_ids, attention_mask=attention_mask, labels=answers)
        
        ext_out.loss.backward()

        gen_out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        # TODO: gen_out.loss.backward()

        ext_losses.append(ext_out.loss.item())
        gen_losses.append(gen_out.loss.item())

        if (i+1) % backward_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        i += 1

    # epoch이 끝나면 누적 loss 데이터 전체 저장
    with open(f"./ext_losses_ep_{epoch}.pkl", "wb") as f:
        pickle.dump(ext_losses, f)
    with open(f"./gen_losses_ep_{epoch}.pkl", "wb") as f:
        pickle.dump(gen_losses, f)

model.save_pretrained("./saved")

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot()
# interval = 100
# ext_mean = np.array(ext_losses)[:(len(ext_losses)//interval) * interval].reshape(interval, -1).mean(1)
# gen_mean = np.array(gen_losses)[:(len(ext_losses)//interval) * interval].reshape(interval, -1).mean(1)

# plt.plot(ext_mean)
# plt.plot(gen_mean)
