import sys
from os import path
sys.path.insert(0, path.abspath('..'))
from test.testftns import TestLasttoken, Lambada, ReadJson
from fquant import FakeQuantize
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

path = "results_fquant_int8.txt"
#dataset_lambada = Lambada(batch_size=32)
openai_lambada = ReadJson("/home/cid2/dataset/lambada_test.jsonl", batch_size=32)

tokenizer_small = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_small_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
gpt2_small_model.half()

gpt2_fake_quant_model = FakeQuantize(gpt2_small_model)
TestLasttoken(gpt2_fake_quant_model, tokenizer_small, openai_lambada, verbose = False, comment = "INT8 Fake Quantized GPT-2 small openai lambada", path = path)

gpt2_fake_quant_model = FakeQuantize(gpt2_small_model, LastLinear = True)
TestLasttoken(gpt2_fake_quant_model, tokenizer_small, openai_lambada, verbose = False, comment = "INT8 Fake Quantized GPT-2 small + output linear layer openai lambada", path = path)

gpt2_fake_quant_model = FakeQuantize(gpt2_small_model, kbits = [[4]*4 for _ in range(12)])
TestLasttoken(gpt2_fake_quant_model, tokenizer_small, openai_lambada, verbose = False, comment = "INT4 Fake Quantized GPT-2 small openai lambada", path = path)

tokenizer_xl = GPT2Tokenizer.from_pretrained('gpt2-xl')
gpt2_xl_model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to(device)
gpt2_xl_model.half()

gpt2_fake_quant_model = FakeQuantize(gpt2_xl_model)
TestLasttoken(gpt2_fake_quant_model, tokenizer_xl, openai_lambada, verbose = False, comment = "INT8 Fake Quantized GPT-2 xl openai lambada", path = path)

gpt2_fake_quant_model = FakeQuantize(gpt2_xl_model, LastLinear = True)
TestLasttoken(gpt2_fake_quant_model, tokenizer_xl, openai_lambada, verbose = False, comment = "INT8 Fake Quantized GPT-2 xl + output linear layer openai lambada", path = path)

gpt2_fake_quant_model = FakeQuantize(gpt2_xl_model, kbits = [[4]*4 for _ in range(48)])
TestLasttoken(gpt2_fake_quant_model, tokenizer_xl, openai_lambada, verbose = False, comment = "INT4 Fake Quantized GPT-2 xl openai lambada", path = path)
