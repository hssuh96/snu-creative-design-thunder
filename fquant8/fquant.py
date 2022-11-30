import torch
from torch import nn
import torch.nn.functional as F
import copy

class QuantConv1d(nn.Module):

    def __init__(self, conv1d, k = 8, linear = False):
        super(QuantConv1d, self).__init__()
        twok = (1<<(k-1))
        if linear:
            w0 = torch.transpose(conv1d.weight.data, 0, 1)
            self.nf = conv1d.out_features
        else:
            w0 = conv1d.weight.data
            self.nf = conv1d.nf
        m, _ = torch.min(w0, dim = 0)
        M, _ = torch.max(w0, dim = 0)
        s = torch.where(M < -m, m, M)/twok
        s = s.reshape(1, -1)
        w = torch.round((torch.clamp(w0/s.expand(w0.size(0), -1), -twok, twok-1))).type(torch.int8)
        self.scale = nn.Parameter(s)
        self.weight = nn.Parameter(w, requires_grad = False)
        if conv1d.bias != None:
            self.bias = nn.Parameter(conv1d.bias.clone().detach())
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.mm(x.view(-1, x.size(-1)), self.weight.type(x.dtype))
        x = x * self.scale.expand(x.size(0), -1)
        if self.bias != None:
            x = torch.add(self.bias, x)
        x = x.view(size_out)
        return x

class QuantLinear(nn.Module):
    def __init__(self, linear, k = 8):
        super(QuantLinear, self).__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        twok = (1<<(k-1))
        w0 = torch.transpose(linear.weight.data, -2, -1)
        m, _ = torch.min(w0, dim = 0)
        M, _ = torch.max(w0, dim = 0)
        s = torch.where(M<-m, m, M)/twok
        s = s.reshape(1, -1)
        w = torch.round((torch.clamp(w0/s.expand(w0.size(0), -1), -twok, twok-1))).type(torch.int8)
        self.scale = nn.Parameter(s)
        self.weight = nn.Parameter(w, requires_grad = False)
        if linear.bias:
            self.bias = nn.Parameter(linear.bias.clone().detach())
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.out_features,)
        x = torch.mm(x.view(-1, x.size(-1)), self.weight.type(x.dtype))
        x = x * self.scale.expand(x.size(0), -1)
        if self.bias:
            x = torch.add(self.bias, x)
        x = x.view(size_out)
        return x



def FakeQuantize(model, kbits = None, LastLinear = False):
    qmodel = copy.deepcopy(model)
    if kbits == None:
        kbits = [[8] * 4 for _ in range(len(qmodel.transformer.h))]
    for i, block in enumerate(qmodel.transformer.h):
        attn = block.attn
        if attn.is_cross_attention:
            attn.q_attn = QuantConv1d(attn.q_attn, kbits[i][5])
        attn.c_attn = QuantConv1d(attn.c_attn, kbits[i][0])
        attn.c_proj = QuantConv1d(attn.c_proj, kbits[i][1])
        mlp = block.mlp
        mlp.c_fc = QuantConv1d(mlp.c_fc, kbits[i][2])
        mlp.c_proj = QuantConv1d(mlp.c_proj, kbits[i][3])
    if LastLinear:
        qmodel.lm_head = QuantConv1d(qmodel.lm_head, linear = True)
    return qmodel


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
