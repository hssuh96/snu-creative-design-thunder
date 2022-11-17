import torch
from torch import nn

class QuantConv1d(nn.Module):

    def __init__(self, conv1d, k = 8):
        super().__init__()
        self.nf = conv1d.nf
        twok = (1<<(k-1))
        w0 = conv1d.weight.data
        m, _ = torch.min(w0, dim = 0)
        M, _ = torch.max(w0, dim = 0)
        s = torch.where(M < -m, m, M)/twok
        s = s.reshape(1, -1)
        w = torch.round((torch.clamp(w0/s.expand(w0.size(0), -1), -twok, twok-1))).type(torch.int8)
        self.scale = nn.Parameter(s)
        self.weight = nn.Parameter(w, requires_grad = False)
        self.bias = nn.Parameter(conv1d.bias.clone().detach())
    
    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.mm(x.view(-1, x.size(-1)), self.weight.type(x.dtype))
        x = x * self.scale.expand(x.size(0), -1)
        x = torch.add(self.bias, x)
        x = x.view(size_out)
        return x

def FakeQuantize(model):
    for block in model.transformer.h:
        attn = block.attn
        if attn.is_cross_attention:
            attn.q_attn = QuantConv1d(attn.q_attn)
        attn.c_attn = QuantConv1d(attn.c_attn)
        attn.c_proj = QuantConv1d(attn.c_proj)
        mlp = block.mlp
        mlp.c_fc = QuantConv1d(mlp.c_fc)
        mlp.f_proj = QuantConv1d(mlp.c_proj)
    return model


'''
class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
'''
