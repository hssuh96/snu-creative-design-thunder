from testftns import TestLasttoken
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def quantize(x, k):
    row_max, _ = torch.max(torch.abs(x), dim=0)
    twok = (1<<k)-1
    return (torch.round((x/row_max+1)/2*twok)/twok*2-1)*row_max

lambada = load_dataset("lambada", split="test")
lines = []
for line in lambada:
    lines.append(line['text'])
lambada_loader = DataLoader(lines, batch_size = 16, shuffle = False)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_base_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

model_quant8 = copy.deepcopy(gpt2_base_model)
for name, param in model_quant8.named_parameters():
    if "c_attn.weight" in name or "c_proj.weight" in name or "c_fc.weight" in name:
        param.data=quantize(param.data, 8)

TestLasttoken(model_quant8, tokenizer, lambada_loader)

model_quant4 = copy.deepcopy(gpt2_base_model)
for name, param in model_quant4.named_parameters():
    if "c_attn.weight" in name or "c_proj.weight" in name or "c_fc.weight" in name:
        param.data=quantize(param.data, 4)

TestLasttoken(model_quant4, tokenizer, lambada_loader)

