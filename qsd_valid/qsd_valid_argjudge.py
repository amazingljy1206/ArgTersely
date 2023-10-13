import torch
import numpy as np

model = torch.load('filtering/models/bert-base-finetuned.pt')

with open('qsd_valid/qsd.txt', 'r', encoding = 'UTF-8') as f:
    lines = f.readlines()

res = []
for line in lines:
    ag, sent1, sent2 = line.strip().split('\t')
    sent2 = sent2.strip('"')
    res.append(model.inference(ag, [sent1, sent2]))
# print([i for i, x in enumerate(res) if x == 1])
print(np.mean(res))

