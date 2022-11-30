import sys
from os import path
sys.path.insert(0, path.abspath('..'))
from test.testftns import TestLasttoken, Lambada, ReadJson
from bquant import BQGPT2
from bquant import BQuantConv1d_simple2 as BQuantConv1d
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
device = "cuda:3" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

path = "results_bquant.txt"
#dataset_lambada = Lambada(batch_size=32)
openai_lambada = ReadJson("/home/cid2/dataset/lambada_test.jsonl", batch_size=32)

tokenizer_xl = GPT2Tokenizer.from_pretrained('gpt2-xl')
gpt2_xl_model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to(device)
gpt2_xl_model.half()

gpt2_fake_quant_model = BQGPT2(gpt2_xl_model, BQuantConv1d)
TestLasttoken(gpt2_fake_quant_model, tokenizer_xl, openai_lambada, verbose = False, comment = "8-bit greedy supnorm binary Quantized simple2 GPT-2 xl openai lambada", path = path)
gpt2_fake_quant_model.setbits([[6]*4]*48)
TestLasttoken(gpt2_fake_quant_model, tokenizer_xl, openai_lambada, verbose = False, comment = "6-bit greedy supnorm binary Quantized simple2 GPT-2 xl openai lambada", path = path)
gpt2_fake_quant_model.setbits([[4]*4]*48)
TestLasttoken(gpt2_fake_quant_model, tokenizer_xl, openai_lambada, verbose = False, comment = "4-bit greedy supnorm binary Quantized simple2 GPT-2 xl openai lambada", path = path)

#gpt2_fake_quant_model = BinaryQuantize(gpt2_xl_model, kbits = [[4]*4 for _ in range(48)])
#TestLasttoken(gpt2_fake_quant_model, tokenizer_xl, openai_lambada, verbose = False, comment = "4-bit greedy supnorm binary Quantized GPT-2 xl openai lambada", path = path)
