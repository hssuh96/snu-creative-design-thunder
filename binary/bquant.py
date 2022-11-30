import torch
from torch import nn
import torch.nn.functional as F
import copy
from transformers import GPT2LMHeadModel

class BQuantConv1d_csr(nn.Module):

    def __init__(self, conv1d, k = 8, method = 'sup', linear = False):
        super(BQuantConv1d_csr, self).__init__()

        twok = (1<<(k-1))
        if linear:
            w0 = conv1d.weight.data
            self.nf = conv1d.out_features
            self.nx = conv1d.in_features
        else:
            w0 = torch.transpose(conv1d.weight.data, 0, 1)
            self.nf = conv1d.nf
            self.nx = w0.size(1)
        n, m = w0.size()
        self.bits = k

        binary_list = []
        scale_list = []

        binary_exp = torch.tensor([2**k for k in range(7, -1, -1)], device = w0.device, dtype=w0.dtype).view(1, -1, 1)
        self.map4 = torch.tensor([[1 if ((j>>i)&1)!=0 else -1 for j in range(16)] for i in range(3, -1, -1)], device = w0.device, dtype = w0.dtype)

        for _ in range(self.bits):
            absw0 = torch.abs(w0)
            sign = w0>0
            if method == 'sup':
                mini, _ = torch.min(absw0, dim = 1)
                maxi, _ = torch.max(absw0, dim = 1)
                s = (mini+maxi)/2
                s = s.view(-1, 1)
                scale_list.append(s)
            elif method == 'l2':
                s = torch.mean(absw0, dim = 1).view(-1, 1)
                scale_list.append(s)
            b = torch.where(sign, 1, 0).type(w0.dtype).view(n, m//8, 8)
            b = torch.matmul(b, binary_exp).squeeze(2).type(torch.uint8)
            binary_list.append(b)
            w0 = w0 - torch.where(sign, s.expand(-1, m), -s.expand(-1, m))

        s = torch.stack(scale_list, dim = 0)
        B = torch.stack(binary_list, dim = 0).to(w0.device)
        self.scale = nn.Parameter(s)
        self.binary = nn.Parameter(B, requires_grad = False)

        if conv1d.bias != None:
            self.bias = nn.Parameter(conv1d.bias.clone().detach())
        else:
            self.register_parameter('bias', None)

        #self.binary (k, n, m/8)
        #self.scale (k, n, 1)
        #self.bias (n)
        #self.bits scalar

    def MakeTable(self, x):
        y = x.view(-1, x.size(1)//8, 8)
        y0 = y[:, :, 0:4]
        y1 = y[:, :, 4:8]
        subtable0 = torch.matmul(y0, self.map4).unsqueeze(3)
        subtable1 = torch.matmul(y1, self.map4).unsqueeze(2)

        table = subtable0.expand(-1, -1, -1, 16) + subtable1.expand(-1, -1, 16, -1)
        table = table.view(table.size(0), -1)
        table = table.transpose(0, 1)
        #table = table.reshape(table.size(0), -1, 256)
        return table
    
    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = x.view(-1, x.size(-1))
        table = self.MakeTable(x).type(torch.float32)


        #ind1 = self.binary[:, 0:self.bits].long()+torch.arange(0, 256*self.nx//8, 256, dtype = torch.long, device = x.device).view(1, 1, -1, 1)
        #ind1 = ind1.flatten()
        #ind = torch.stack([torch.arange(0, self.nf, dtype = torch.long, device = x.device).repeat(self.bits*self.nx//8), ind1], dim = 0)
        offset = torch.arange(0, 256*self.nx//8, 256, device = x.device, dtype = torch.int).view(1, -1)
        crow = torch.arange(0, self.nx//8*(self.nf+1), self.nx//8, dtype = torch.int, device = x.device)
        out_list = []
        for i in range(self.bits):
            sp = torch.sparse_csr_tensor(crow, (self.binary[i].int() + offset).flatten(), self.scale[i].repeat(1, self.nx//8).flatten(), size = (self.nf, self.nx//8*256), dtype=torch.float32, device = x.device)
            out_list.append(torch.sparse.mm(sp, table))
            #sparse_list.append(sp)
            
        #sparse = torch.stack(sparse_list, dim = 0)
        #sparse = torch.sum(sparse, dim = 0)
        #col = self.binary[0:self.bits].int() + torch.arange(0, 256*self.nx//8, 256, device = x.device, dtype = torch.int).view(1, 1, -1)
        #col = col.view(self.bits, -1)
        #sparse = torch.sparse_csr_tensor(crow, col, torch.ones((self.bits, self.nx*self.nf//8), device = x.device), size = (self.bits, self.nf, self.nx//8*256), dtype = torch.float32, device = x.device)
        

        #print(sparse.size(), table.size())
        #out = torch.matmul(sparse, table.type(torch.float32)).type(self.scale.dtype)
        out = torch.sum(torch.stack(out_list, dim = 0), dim = 0)
        
        #out = out*self.scale[0:self.bits]
        #out = torch.sum(out, dim = 0)
        
        out = out.transpose(0, 1).type(self.scale.dtype)

        if self.bias!=None:
            out = torch.add(out, self.bias)
        out = out.view(size_out)
        return out

class BQuantConv1d_coo(nn.Module):

    def __init__(self, conv1d, k = 8, method = 'sup', linear = False):
        super(BQuantConv1d_coo, self).__init__()

        twok = (1<<(k-1))
        if linear:
            w0 = conv1d.weight.data
            self.nf = conv1d.out_features
            self.nx = conv1d.in_features
        else:
            w0 = torch.transpose(conv1d.weight.data, 0, 1)
            self.nf = conv1d.nf
            self.nx = w0.size(1)
        n, m = w0.size()
        self.bits = k

        binary_list = []
        scale_list = []

        binary_exp = torch.tensor([2**k for k in range(7, -1, -1)], device = w0.device, dtype=w0.dtype).view(1, -1, 1)
        self.map4 = torch.tensor([[1 if ((j>>i)&1)!=0 else -1 for j in range(16)] for i in range(3, -1, -1)], device = w0.device, dtype = w0.dtype)

        for _ in range(self.bits):
            absw0 = torch.abs(w0)
            sign = w0>0
            if method == 'sup':
                mini, _ = torch.min(absw0, dim = 1)
                maxi, _ = torch.max(absw0, dim = 1)
                s = (mini+maxi)/2
                s = s.view(-1, 1)
                scale_list.append(s)
            elif method == 'l2':
                s = torch.mean(absw0, dim = 1).view(-1, 1)
                scale_list.append(s)
            b = torch.where(sign, 1, 0).type(w0.dtype).view(n, m//8, 8)
            b = torch.matmul(b, binary_exp).squeeze(2).type(torch.uint8)
            binary_list.append(b)
            w0 = w0 - torch.where(sign, s.expand(-1, m), -s.expand(-1, m))

        s = torch.stack(scale_list, dim = 0)
        B = torch.stack(binary_list, dim = 0).to(w0.device)
        self.scale = nn.Parameter(s)
        self.binary = nn.Parameter(B, requires_grad = False)

        if conv1d.bias != None:
            self.bias = nn.Parameter(conv1d.bias.clone().detach())
        else:
            self.register_parameter('bias', None)

        #self.binary (k, n, m/8)
        #self.scale (k, n, 1)
        #self.bias (n)
        #self.bits scalar

    def MakeTable(self, x):
        y = x.view(-1, x.size(1)//8, 8)
        y0 = y[:, :, 0:4]
        y1 = y[:, :, 4:8]
        subtable0 = torch.matmul(y0, self.map4).unsqueeze(3)
        subtable1 = torch.matmul(y1, self.map4).unsqueeze(2)

        table = subtable0.expand(-1, -1, -1, 16) + subtable1.expand(-1, -1, 16, -1)
        table = table.view(table.size(0), -1)
        table = table.transpose(0, 1)
        #table = table.reshape(table.size(0), -1, 256)
        return table
    
    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = x.view(-1, x.size(-1))
        table = self.MakeTable(x).type(torch.float32)


        ind1 = self.binary[0:self.bits].long()+torch.arange(0, 256*self.nx//8, 256, dtype = torch.long, device = x.device).view(1, 1, -1)
        ind1 = ind1.flatten()
        ind = torch.stack([torch.arange(0, self.nf, dtype = torch.long, device = x.device).view(1, -1, 1).expand(self.bits, -1, self.nx//8).flatten(), ind1], dim = 0)
        #offset = torch.arange(0, 256*self.nx//8, 256, device = x.device, dtype = torch.int).view(1, -1)
        sparse = torch.sparse_coo_tensor(ind, self.scale.expand(-1, -1, self.nx//8).flatten(), (self.nf, self.nx//8*256), dtype = torch.float32, device = x.device) 

        #print(sparse.size(), table.size())
        out = torch.sparse.mm(sparse, table).type(self.scale.dtype)
        
        #out = out*self.scale[0:self.bits]
        #out = torch.sum(out, dim = 0)
        
        out = out.transpose(0, 1)

        if self.bias!=None:
            out = torch.add(out, self.bias)
        out = out.reshape(size_out)
        return out

class BQuantConv1d_cpu1(nn.Module):

    def __init__(self, conv1d, k = 8, method = 'sup', linear = False):
        super(BQuantConv1d_cpu1, self).__init__()

        twok = (1<<(k-1))
        if linear:
            w0 = torch.transpose(conv1d.weight.data, 0, 1)
            self.nf = conv1d.out_features
            self.nx = conv1d.in_features
        else:
            w0 = conv1d.weight.data
            self.nf = conv1d.nf
            self.nx = w0.size(0)
        m, n = w0.size()
        self.bits = k

        binary_list = []
        scale_list = []

        self.binary_exp = torch.tensor([2**k for k in range(7, -1, -1)], device = w0.device, dtype=w0.dtype).view(1, 1, -1)
        self.map4 = torch.tensor([[1 if ((j>>i)&1)!=0 else -1 for j in range(16)] for i in range(3, -1, -1)], device = w0.device, dtype = w0.dtype)

        for _ in range(self.bits):
            absw0 = torch.abs(w0)
            sign = w0>0
            if method == 'sup':
                mini, _ = torch.min(absw0, dim = 0)
                maxi, _ = torch.max(absw0, dim = 0)
                s = (mini+maxi)/2
                scale_list.append(s)
            elif method == 'l2':
                s = torch.mean(absw0, dim = 0)
                scale_list.append(s)
            b = torch.where(sign, 1, 0).type(w0.dtype).view(m//8, 8, n)
            b = torch.matmul(self.binary_exp, b).squeeze(1).type(torch.uint8)
            binary_list.append(b)
            w0 = w0 - torch.where(sign, s.expand(m, -1), -s.expand(m, -1))

        s = torch.stack(scale_list, dim = 0).unsqueeze(0)
        B = torch.stack(binary_list, dim = 0).unsqueeze(0).to(w0.device)
        self.scale = nn.Parameter(s)
        self.binary = nn.Parameter(B, requires_grad = False)

        if conv1d.bias != None:
            self.bias = nn.Parameter(conv1d.bias.clone().detach())
        else:
            self.register_parameter('bias', None)

        #self.binary (1, k, m/8, n)
        #self.scale (1, k, n)
        #self.bias (n)
        #self.bits scalar

    def MakeTable(self, x):
        y = x.view(-1, x.size(1)//8, 8)
        y0 = y[:, :, 0:4]
        y1 = y[:, :, 4:8]
        subtable0 = torch.matmul(y0, self.map4).unsqueeze(3)
        subtable1 = torch.matmul(y1, self.map4).unsqueeze(2)

        table = subtable0.expand(-1, -1, -1, 16) + subtable1.expand(-1, -1, 16, -1)
        table = table.reshape(table.size(0), 1, -1, 256)
        #table = table.reshape(table.size(0), -1, 256)
        return table
    
    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = x.view(-1, x.size(-1))
        table = self.MakeTable(x)
        #ind = self.binary[:, 0:self.bits].long().expand(x.size(0), -1, -1, -1)
        ind = self.binary[:, :self.bits].long().squeeze(0)
        out = torch.zeros((x.size(0), self.bits, self.nf), dtype = x.dtype, device = x.device)
        table = table.expand(-1, self.bits, -1, -1)
        for i in range(0, self.nx//8):
            #out += torch.gather(table[:, :, i, :], dim = 2, index = self.binary[:, :self.bits, i, :].long().expand(x.size(0), -1, -1))
            #out += table[:, torch.arange(self.bits)[:, None], i, ind[:, :, i]]
            #out += (table[:, :, i, :])[(torch.arange(self.bits)[:, None]), (ind[:, :,  i])]
            out += table[:, 0, i, ind[:, i, :]]
            #out += torch.index_select(table[:, :, i, :], dim = 2, index = ind[:, :, i, :])
        #out = torch.gather(table.expand(-1, self.bits, -1, -1), dim = 3, index = ind)
        #out = torch.sum(out, dim = 2)

        out = out * self.scale[:, 0:self.bits, :]
        out = torch.sum(out, dim = 1)
        # out = torch.bmm(torch.permute(out, (2, 0, 1)), self.scale[:, 0:self.bits, :]) #scale.shape = (n, k, 1)
        # out = torch.permute(out, (1, 0)).contiguous()
        
        if self.bias!=None:
            out = torch.add(out, self.bias)
        out = out.view(size_out)
        return out

class BQuantConv1d_toobig(nn.Module):

    def __init__(self, conv1d, k = 8, method = 'sup', linear = False):
        super(BQuantConv1d_toobig, self).__init__()

        twok = (1<<(k-1))
        if linear:
            w0 = torch.transpose(conv1d.weight.data, 0, 1)
            self.nf = conv1d.out_features
            self.nx = conv1d.in_features
        else:
            w0 = conv1d.weight.data
            self.nf = conv1d.nf
            self.nx = w0.size(0)
        m, n = w0.size()
        self.bits = k

        binary_list = []
        scale_list = []

        binary_exp = torch.tensor([2**k for k in range(7, -1, -1)], device = w0.device, dtype=w0.dtype).view(1, 1, -1)
        self.map4 = torch.tensor([[1 if ((j>>i)&1)!=0 else -1 for j in range(16)] for i in range(3, -1, -1)], device = w0.device, dtype = w0.dtype)

        for _ in range(self.bits):
            absw0 = torch.abs(w0)
            sign = w0>0
            if method == 'sup':
                mini, _ = torch.min(absw0, dim = 0)
                maxi, _ = torch.max(absw0, dim = 0)
                s = (mini+maxi)/2
                scale_list.append(s)
            elif method == 'l2':
                s = torch.mean(absw0, dim = 0)
                scale_list.append(s)
            b = torch.where(sign, 1, 0).type(w0.dtype).view(m//8, 8, n)
            b = torch.matmul(binary_exp, b).squeeze(1).type(torch.uint8)
            binary_list.append(b)
            w0 = w0 - torch.where(sign, s.expand(m, -1), -s.expand(m, -1))

        s = torch.stack(scale_list, dim = 0).unsqueeze(0)
        B = torch.stack(binary_list, dim = 0).unsqueeze(0).to(w0.device)
        self.scale = nn.Parameter(s)
        self.binary = nn.Parameter(B, requires_grad = False)

        if conv1d.bias != None:
            self.bias = nn.Parameter(conv1d.bias.clone().detach())
        else:
            self.register_parameter('bias', None)

        #self.binary (1, k, m/8, n)
        #self.scale (1, k, n)
        #self.bias (n)
        #self.bits scalar

    def MakeTable(self, x):
        y = x.view(-1, x.size(1)//8, 8)
        y0 = y[:, :, 0:4]
        y1 = y[:, :, 4:8]
        subtable0 = torch.matmul(y0, self.map4).unsqueeze(3)
        subtable1 = torch.matmul(y1, self.map4).unsqueeze(2)

        table = subtable0.expand(-1, -1, -1, 16) + subtable1.expand(-1, -1, 16, -1)
        table = table.reshape(table.size(0), 1, -1, 256)
        #table = table.reshape(table.size(0), -1, 256)
        return table
    
    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = x.view(-1, x.size(-1))
        table = self.MakeTable(x)
        ind = self.binary[:, 0:self.bits].long().expand(x.size(0), -1, -1, -1)


        out = torch.gather(table.expand(-1, self.bits, -1, -1), dim = 3, index = ind)
        out = torch.sum(out, dim = 2)

        out = out * self.scale[:, 0:self.bits, :]
        out = torch.sum(out, dim = 1)
        # out = torch.bmm(torch.permute(out, (2, 0, 1)), self.scale[:, 0:self.bits, :]) #scale.shape = (n, k, 1)
        # out = torch.permute(out, (1, 0)).contiguous()
        
        if self.bias!=None:
            out = torch.add(out, self.bias)
        out = out.view(size_out)
        return out

class BQuantConv1d_simple(nn.Module):

    def __init__(self, conv1d, k = 8, method = 'sup', linear = False):
        super(BQuantConv1d_simple, self).__init__()

        twok = (1<<(k-1))
        if linear:
            w0 = torch.transpose(conv1d.weight.data, 0, 1)
            self.nf = conv1d.out_features
        else:
            w0 = conv1d.weight.data
            self.nf = conv1d.nf
        m, n = w0.size()
        self.bits = k

        binary_list = []
        scale_list = []

        #binary_exp = torch.tensor([2**k for k in range(7, -1, -1)], device = w0.device, dtype=w0.dtype).view(1, 1, -1)
        #self.map4 = torch.tensor([[1 if ((j>>i)&1)!=0 else -1 for j in range(16)] for i in range(3, -1, -1)], device = w0.device, dtype = w0.dtype)

        if method == 'sup':
            for _ in range(self.bits):
                absw0 = torch.abs(w0)
                sign = w0>0
                mini, _ = torch.min(absw0, dim = 0)
                maxi, _ = torch.max(absw0, dim = 0)
                s = (mini+maxi)/2
                scale_list.append(s)
                b = torch.where(sign, 1, -1).type(torch.int8)
                binary_list.append(b)
                w0 = w0 - torch.where(sign, s.expand(m, -1), -s.expand(m, -1))
        elif method == 'l2':
            for _ in range(self.bits):
                absw0 = torch.abs(w0)
                sign = w0>0
                s = torch.mean(absw0, dim = 0)
                scale_list.append(s)
                b = torch.where(sign, 1, -1).type(torch.int8)
                binary_list.append(b)
                w0 = w0 - torch.where(sign, s.expand(m, -1), -s.expand(m, -1))

        s = torch.stack(scale_list, dim = 0).unsqueeze(1)
        B = torch.stack(binary_list, dim = 0).to(w0.device)
        self.scale = nn.Parameter(s)
        self.binary = nn.Parameter(B, requires_grad = False)

        if conv1d.bias != None:
            self.bias = nn.Parameter(conv1d.bias.clone().detach())
        else:
            self.register_parameter('bias', None)

    def forward(self, x): 
        size_out = x.size()[:-1] + (self.nf,)
        x = x.view(1, -1, x.size(-1))
        x = torch.matmul(x, self.binary[0:self.bits].type(x.dtype))
        x = x * self.scale[0:self.bits].expand(-1, x.size(1), -1)
        x = torch.sum(x, dim = 0)
        if self.bias!=None:
            x = torch.add(x, self.bias)
        x = x.view(size_out)
        return x

class BQuantConv1d_simple2(nn.Module):

    def __init__(self, conv1d, k = 8, method = 'sup', linear = False):
        super(BQuantConv1d_simple2, self).__init__()

        twok = (1<<(k-1))
        if linear:
            w0 = torch.transpose(conv1d.weight.data, 0, 1)
        else:
            w0 = conv1d.weight.data
        m, n = w0.size()
        self.nf = n
        self.nx = m
        self.bits = k

        binary_list = []
        scale_list = []

        self.binary_exp = torch.tensor([2**k for k in range(7, -1, -1)], device = w0.device, dtype=w0.dtype).view(1, 1, -1)
        #self.map4 = torch.tensor([[1 if ((j>>i)&1)!=0 else -1 for j in range(16)] for i in range(3, -1, -1)], device = w0.device, dtype = w0.dtype)

        if method == 'sup':
            for _ in range(self.bits):
                absw0 = torch.abs(w0)
                sign = w0>0
                mini, _ = torch.min(absw0, dim = 0)
                maxi, _ = torch.max(absw0, dim = 0)
                s = (mini+maxi)/2
                scale_list.append(s)
                b = torch.where(sign, 1.0, 0.0).to(w0.device, w0.dtype)
                binary_list.append(b)
                w0 = w0 - torch.where(sign, s.expand(m, -1), -s.expand(m, -1))
        elif method == 'l2':
            for _ in range(self.bits):
                absw0 = torch.abs(w0)
                sign = w0>0
                s = torch.mean(absw0, dim = 0)
                scale_list.append(s)
                b = torch.where(sign, 1.0, 0.0).to(w0.device, w0.dtype)
                binary_list.append(b)
                w0 = w0 - torch.where(sign, s.expand(m, -1), -s.expand(m, -1))


        s = torch.stack(scale_list, dim = 1).unsqueeze(1)
        B = torch.stack(binary_list, dim = 0).view(self.bits, m//8, 8, n)
        B = torch.matmul(self.binary_exp, B).type(torch.int8).permute(3, 0, 1, 2).contiguous() #(n, k, m//8, 1)

        self.scale = nn.Parameter(s)
        self.binary = nn.Parameter(B, requires_grad = False)

        if conv1d.bias != None:
            self.bias = nn.Parameter(conv1d.bias.clone().detach())
        else:
            self.register_parameter('bias', None)


    def forward(self, x): 
        size_out = x.size()[:-1] + (self.nf,)
        x = x.view(-1, x.size(-1))
        b = self.binary[:, :self.bits].expand(-1, -1, -1, 8)
        weight = torch.where(torch.bitwise_and(b, self.binary_exp.view(1, 1, 1, -1).type(torch.int8))==0, -1, 1).to(x.device, x.dtype)
        weight = weight.view(self.nf, self.bits, self.nx)
        weight = torch.bmm(self.scale[:,:,:self.bits], weight).squeeze(1).permute(1, 0)
        x = torch.addmm(self.bias, x, weight)
        x = x.view(size_out)
        return x

class BQuantConv1d_scatter_add(nn.Module):

    def __init__(self, conv1d, k = 8, method = 'sup', linear = False):
        super(BQuantConv1d_scatter_add, self).__init__()

        twok = (1<<(k-1))
        if linear:
            w0 = torch.transpose(conv1d.weight.data, 0, 1)
        else:
            w0 = conv1d.weight.data
        m, n = w0.size()
        self.nf = n
        self.nx = m
        self.bits = k

        binary_list = []
        scale_list = []

        self.binary_exp = torch.tensor([2**k for k in range(7, -1, -1)], device = w0.device, dtype=w0.dtype).view(1, 1, -1)
        #self.map4 = torch.tensor([[1 if ((j>>i)&1)!=0 else -1 for j in range(16)] for i in range(3, -1, -1)], device = w0.device, dtype = w0.dtype)

        if method == 'sup':
            for _ in range(self.bits):
                absw0 = torch.abs(w0)
                sign = w0>0
                mini, _ = torch.min(absw0, dim = 0)
                maxi, _ = torch.max(absw0, dim = 0)
                s = (mini+maxi)/2
                scale_list.append(s)
                b = torch.where(sign, 1.0, 0.0).to(w0.device, w0.dtype)
                binary_list.append(b)
                w0 = w0 - torch.where(sign, s.expand(m, -1), -s.expand(m, -1))
        elif method == 'l2':
            for _ in range(self.bits):
                absw0 = torch.abs(w0)
                sign = w0>0
                s = torch.mean(absw0, dim = 0)
                scale_list.append(s)
                b = torch.where(sign, 1.0, 0.0).to(w0.device, w0.dtype)
                binary_list.append(b)
                w0 = w0 - torch.where(sign, s.expand(m, -1), -s.expand(m, -1))


        s = torch.stack(scale_list, dim = 1).unsqueeze(2)
        B = torch.stack(binary_list, dim = 0).view(self.bits, m//8, 8, n)
        B = torch.matmul(self.binary_exp, B).type(torch.int8).permute(3, 0, 1, 2).contiguous()

        self.scale = nn.Parameter(s)
        self.binary = nn.Parameter(B, requires_grad = False)

        if conv1d.bias != None:
            self.bias = nn.Parameter(conv1d.bias.clone().detach())
        else:
            self.register_parameter('bias', None)


    def forward(self, x): 
        size_out = x.size()[:-1] + (self.nf,)
        x = x.view(-1, x.size(-1))
        b = self.binary[:, :self.bits].expand(-1, -1, -1, 8)
        weight = torch.bitwise_and(b, self.binary_exp.view(1, 1, 1, -1).type(torch.int8)).ne(0).to(x.device, torch.long)
        weight = weight.view(self.nf, 1, self.bits, self.nx).expand(-1, x.size(0), -1, -1)
        y = x.view(1, x.size(0), 1, -1).expand(self.nf, -1, self.bits, -1)
        summed = torch.zeros((self.nf, x.size(0), self.bits, 2), device = x.device, dtype = x.dtype).scatter_(3, weight, y, reduce = 'add')
        out = summed[:, :, :, 1]-summed[:, :, :, 0]
        out = torch.matmul(out, self.scale[:, :, :self.bits]).squeeze(2).permute(1, 0)
        out = torch.add(self.bias, out)
        out = out.view(size_out)
        return out

class BQGPT2(nn.Module):
    def __init__(self, model, qlayer, method = 'sup', kbits = None, LastLinear = False):
        super(BQGPT2, self).__init__()
        self.gpt2 = copy.deepcopy(model)
        self.device = model.device
        if kbits == None:
            kbits = [[8]*4]*len(self.gpt2.transformer.h)
        for i, block in enumerate(self.gpt2.transformer.h):
            attn = block.attn
            if attn.is_cross_attention:
                attn.q_attn = qlayer(attn.q_attn, kbits[i][4], method)
            attn.c_attn = qlayer(attn.c_attn, kbits[i][0], method)
            attn.c_proj = qlayer(attn.c_proj, kbits[i][1], method)
            mlp = block.mlp
            mlp.c_fc = qlayer(mlp.c_fc, kbits[i][2], method)
            mlp.c_proj = qlayer(mlp.c_proj, kbits[i][3], method)
        if LastLinear:
            self.gpt2.lm_head = qlayer(qmodel.lm_head, 8, method, linear = True)

    def forward(self, **args):
        return self.gpt2(**args)

    def setbits(self, kbits):
        for i, block in enumerate(self.gpt2.transformer.h):
            attn = block.attn
            if attn.is_cross_attention:
                attn.q_attn.bits = kbits[i][4]
            attn.c_attn.bits = kbits[i][0]
            attn.c_proj.bits = kbits[i][1]
            mlp = block.mlp
            mlp.c_fc.bits = kbits[i][2]
            mlp.c_proj.bits = kbits[i][3]


def BinaryQuantize(model, kbits = None, LastLinear = False):
    qmodel = copy.deepcopy(model)
    if kbits == None:
        kbits = [[8] * 4 for _ in range(len(qmodel.transformer.h))]
    for i, block in enumerate(qmodel.transformer.h):
        attn = block.attn
        if attn.is_cross_attention:
            attn.q_attn = BQuantConv1d(attn.q_attn, kbits[i][4])
        attn.c_attn = BQuantConv1d(attn.c_attn, kbits[i][0])
        attn.c_proj = BQuantConv1d(attn.c_proj, kbits[i][1])
        mlp = block.mlp
        mlp.c_fc = BQuantConv1d(mlp.c_fc, kbits[i][2])
        mlp.c_proj = BQuantConv1d(mlp.c_proj, kbits[i][3])
    if LastLinear:
        qmodel.lm_head = BQuantConv1d(qmodel.lm_head, linear = True)
    return qmodel


if __name__=='__main__':
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



    conv = Conv1D(6, 16)
    x = torch.randn((5, 16))
    #print(x)
    qconv = BQuantConv1d_cpu1(conv, k= 8, method='sup')
    print(conv(x))
    print(qconv(x))

