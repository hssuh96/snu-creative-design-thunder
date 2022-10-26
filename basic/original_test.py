import sys
from os import path
sys.path.insert(0, path.abspath('..'))
from test.testftns import TestLasttoken, Lambada, ReadJson
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

path = "results_original.txt"
dataset_lambada = Lambada(batch_size=32)
openai_lambada = ReadJson("/home/cid2/dataset/lambada_test.jsonl", batch_size=32)

tokenizer_small = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_small_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

TestLasttoken(gpt2_small_model, tokenizer_small, dataset_lambada, verbose = False, comment = "Original GPT-2 small dataset lambada", path = path)
TestLasttoken(gpt2_small_model, tokenizer_small, openai_lambada, verbose = False, comment = "Original GPT-2 small openai lambada", path = path)

tokenizer_xl = GPT2Tokenizer.from_pretrained('gpt2-xl')
gpt2_xl_model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to(device)

TestLasttoken(gpt2_xl_model, tokenizer_xl, dataset_lambada, verbose = False, comment = "Original GPT-2 xl dataset lambada", path = path)
TestLasttoken(gpt2_xl_model, tokenizer_xl, openai_lambada, verbose = False, comment = "Original GPT-2 xl openai lambada", path = path)

