import sys
from os import path
sys.path.insert(0, path.abspath('..'))
from test.testftns import TestLasttoken, ReadJson
from fquant import FakeQuantize

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).half()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
FakeQuantize(model)
openai_lambada = ReadJson("/home/cid2/dataset/lambada_test.jsonl", batch_size = 64)
TestLasttoken(model, tokenizer, openai_lambada, verbose = True)


