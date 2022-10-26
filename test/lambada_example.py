from testftns import TestLasttoken, Lambada
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

lambada_loader = Lambada()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_base_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

TestLasttoken(gpt2_base_model, tokenizer, lambada_loader, verbose = True, comment = "Original GPT-2 small", path = "result_gpt2small_original.txt")
