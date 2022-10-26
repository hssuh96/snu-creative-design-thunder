from testftns import TestLasttoken
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

lambada = load_dataset("lambada", split="test")
lines = []
for line in lambada:
    lines.append(line['text'])
lambada_loader = DataLoader(lines, batch_size = 16, shuffle = False)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_base_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
gpt2_base_model.half()

TestLasttoken(gpt2_base_model, tokenizer, lambada_loader)
