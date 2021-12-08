import numpy as np

import torch
import torch.nn as nn

import transformers
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, BartForConditionalGeneration, PretrainedConfig
from transformers import BartTokenizerFast
from transformers.models.bart.configuration_bart import BartConfig

from model import BartSummaryModelV2
from utils import collate_fn

print("\n" + "*" * 10)
print("Start!!!")
print("*" * 10 + "\n")

MODEL_NAME = "gogamza/kobart-summarization"

# config = PretrainedConfig.from_pretrained(MODEL_NAME)
# tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
# model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

config = BartConfig.from_pretrained(MODEL_NAME)
tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
model = BartSummaryModelV2.from_pretrained(MODEL_NAME)

print("\n" + "*" * 10)
print("Model initialized")
print("*" * 10 + "\n")

inputs = [
    ["과거를 떠올려보자.", "방송을 보던 우리의 모습을.", "독보적인 매체는 TV였다."],
    ["온 가족이 둘러앉아 TV를 봤다.", "간혹 가족들끼리 뉴스와 드라마, 예능 프로그램을 둘러싸고 리모컨 쟁탈전이 벌어지기도 했다.", "각자 선호하는 프로그램을 ‘본방’으로 보기 위한 싸움이었다.", "TV가 한 대인지 두 대인지 여부도 그래서 중요했다."],
    ["지금은 어떤가.", "‘안방극장’이라는 말은 옛말이 됐다.", "TV가 없는 집도 많다.", "미디어의 혜택을 누릴 수 있는 방법은 늘어났다.", "각자의 방에서 각자의 휴대폰으로, 노트북으로, 태블릿으로 콘텐츠를 즐긴다."]
]
answers = [
    [0, 2],
    [1],
    [0, 3, 4]
]
targets = [
    "과거에 독보적인 매체는 TV였다.",
    "온 가족이 각자 선호하는 프로그램을 보기 위해 리모컨 쟁탈전이 일어났다.",
    "지금은 TV말고도 미디어의 혜택을 누릴 수 있는 방법들이 많아졌다."
]

print("\n" + "*" * 10)
print("Input Preview")
print("*" * 10 + "\n")

print(inputs, answers, targets, sep="\n")

MAX_SEQ_LEN = 32

input_ids = []
attention_mask = []
labels = []

for idx, sentences in enumerate(inputs):
    tokens = [tokenizer.bos_token]
    for sentence in sentences:
        tokens.extend(tokenizer.tokenize(sentence))
        tokens.append(tokenizer.eos_token)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids.append(token_ids)
    attention_mask.append([1] * len(tokens))

    target_tokens = tokenizer.tokenize(targets[idx])
    labels.append([tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(target_tokens) + [tokenizer.eos_token_id])

print("\n" + "*" * 10)
print("Simulating DataLoader")
print("*" * 10 + "\n")

# simulating data loader
inputs = [{"input_ids": input_ids[i], "attention_mask": attention_mask[i], "answers": answers[i], "labels": labels[i]} for i in range(len(input_ids))]
print("inputs before collated:\n", inputs)

inputs = collate_fn(inputs, tokenizer.pad_token_id)
print("inputs collated:\n", inputs)

print("\n" + "*" * 10)
print("Simulating a single step in a train loop")
print("*" * 10 + "\n")

# simulating a single step in a train loop
model.cuda()
model.train()
print("model in train mode (activating dropout)")
print(model)

input_ids = inputs["input_ids"].cuda()
attention_mask = inputs["attention_mask"].cuda()
answers = inputs["answers"].cuda()
labels = inputs["labels"].cuda()

print("\n" + "*" * 10)
print("Extractive Summary")
print("*" * 10 + "\n")

# extractive summary
ext_out = model.classify(input_ids=input_ids, attention_mask=attention_mask, labels=answers)
print("keys of extraction_out:", ext_out.keys())
print("logits:", ext_out.logits)
print("loss:", ext_out.loss)
print("argsort:", ext_out.logits.argsort(dim=1, descending=True))

ext_out.loss.backward()
print("back-propagation completed...")

print("\n" + "*" * 10)
print("Abstractive Summary")
print("*" * 10 + "\n")

# abstractive summary
gen_out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
print("keys of extraction_out:", gen_out.keys())
print("logits:", gen_out.logits)
print("loss:", gen_out.loss)

gen_out.loss.backward()
print("back-propagation completed...")

print("\n" + "*" * 10)
print("Simulating Inference - Abstractive Summary for now")
print("*" * 10 + "\n")

# generate summaries
model.eval()
print("model in eval mode (deactivating dropout)")
summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=8, max_length=48, min_length=16)
print(summary_ids)

print("\n" + "*" * 10)
print("Final decoded results")
print("*" * 10 + "\n")

# decoding
result = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
# "<s>":0,"</s>":1,"<usr>":2,"<pad>":3,"<sys>":4,"<unk>":5,"<mask>":6
# What is <usr> token?
print(result)

print("\n" + "*" * 10)
print("Done")
print("*" * 10 + "\n")