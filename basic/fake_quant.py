import torch
import torch.nn as nn
import copy

def quantize(x, k):
    row_max, _ = torch.max(torch.abs(x), dim=0)
    twok = (1<<k)-1
    return (torch.round((x/row_max+1)/2*twok)/twok*2-1)*row_max

def fake_quantize(model, k):
    model_quant = copy.deepcopy(model)
    for name, param in model_quant.named_parameters():
        if "c_attn.weight" in name or "c_proj.weight" in name or "c_fc.weight" in name:
            param.data=quantize(param.data, k)
    return model_quant
