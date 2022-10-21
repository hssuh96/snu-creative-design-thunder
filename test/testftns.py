import torch
import torch.nn.functional as F

@torch.no_grad()
def TestLasttoken(model, tokenizer, loader):
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
    print(right_num,"/", total_num, right_num/total_num)
    ppl/=total_num
    perplexity = torch.exp(ppl).item()
    print(perplexity)
    return (right_num/total_num, perplexity)