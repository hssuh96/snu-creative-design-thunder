import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

@torch.no_grad()
def _TestLasttoken(model, tokenizer, loader):
    device = model.device
    model.eval()
    total_num = 0
    right_num = 0
    ppl = 0
    for texts in loader:
        tokens = []
        lengths = []
        for text in texts:
            token = tokenizer(text, return_tensors='pt').to(device)
            tokens.append(token)
            length = token['input_ids'].size(1)
            lengths.append(length)
        maxlen = max(lengths)
        input_ids = torch.cat([torch.cat([tokens[i]['input_ids'], torch.zeros((1, maxlen-lengths[i]), dtype=torch.long, device = device)], dim = 1) for i in range(len(tokens))], dim = 0)
        attention_mask = torch.cat([torch.cat([tokens[i]['attention_mask'], torch.zeros((1, maxlen-lengths[i]), dtype=torch.long, device = device)], dim = 1) for i in range(len(tokens))], dim = 0)
        
        input = {'input_ids':input_ids, 'attention_mask':attention_mask}
        output = model(**input)
        gptlogits = torch.gather(output.logits, 1, torch.tensor([x-2 for x in lengths], device=device).view(-1, 1, 1).expand(-1, -1, output.logits.size(2))).squeeze(1)
        gptans = torch.argmax(gptlogits, dim = 1)
        ans = torch.gather(input_ids, 1, torch.tensor([x-1 for x in lengths], device=device).view(-1, 1))
        
        right_num += torch.where(gptans==ans.squeeze(1), 1, 0).sum()
        probs = F.softmax(gptlogits, dim = 1)
        ppl -= torch.log(torch.gather(probs, 1, ans)).sum()
        total_num+=len(texts)

    right_num=right_num.item()
    ppl/=total_num
    perplexity = torch.exp(ppl).item()
    return (right_num/total_num, perplexity)

def getSize(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def TestLasttoken(model, tokenizer, dataloader, verbose = True, comment = None, path = None):
    start_time = time.time()
    acc, ppl = _TestLasttoken(model, tokenizer, dataloader)
    end_time = time.time()
    elapsed_time = end_time-start_time
    size_all_mb = getSize(model)
    
    sizes = "{:.3f}MB".format(size_all_mb)
    accs = "{:.3f}".format(acc*100)
    ppls = "{:.3f}".format(ppl)
    timed = "{:.3f}".format(elapsed_time)
    result_text = (comment if comment else "") + "\nACC: "+str(accs)+"%"+ "\nPPL: "+str(ppls)+ "\nSize: " +str(sizes)+ "\nTime: "+timed+"s\n\n"
    
    
    if verbose:
        print(result_text, end="")
    if path:
        with open(path, 'a') as fp:
            fp.writelines(result_text)


def Lambada(batch_size = 16):
    from datasets import load_dataset
    lambada_dataset = load_dataset("lambada", split="test")
    lines = []
    for data in lambada_dataset:
        lines.append(data['text'])
    loader = DataLoader(lines, batch_size=batch_size, shuffle=False)
    return loader

def ReadJson(path, batch_size=16):
    import jsonlines
    lines = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            lines.append(obj['text'])
    loader = DataLoader(lines, batch_size=batch_size, shuffle=False)
    return loader

