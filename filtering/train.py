import time
import torch
from bert_model import RM_model
import numpy as np
from torch.optim import Adam, sgd
from tqdm import trange
from utils import load_data

device = torch.device('cuda')
epochs = 2
batch_size = 50
lr = 1e-5

train_data, train_data_size = load_data(path = './filtering/rd/train.txt', batch_size = batch_size, shuffle = True)
valid_data, valid_data_size = load_data(path = './filtering/rd/test.txt', batch_size = 10, shuffle = False)
 
model = RM_model(device)
optimizer = Adam(model.parameters(), lr = lr)

begin = time.time()
print('is training model...')
for epoch in trange(epochs, desc = 'Epoch'):
    this_epoch_loss = 0
    model.train()
    for i in trange(len(train_data), desc = 'Batch'):
        loss = 0
        for j in range(batch_size):
            arg  = train_data[i][j]['arg']
            hyps = train_data[i][j]['hyps']
            loss += model(arg, hyps)
        this_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        model.zero_grad()
    this_epoch_loss = this_epoch_loss/train_data_size
    print('\n training loss:{:.2f}'.format(this_epoch_loss))

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for i in range(len(valid_data)):
            loss = 0
            for j in range(10):
                arg  = valid_data[i][j]['arg']
                hyps = valid_data[i][j]['hyps']
                loss += model(arg = arg, hyps = hyps)
            valid_loss += loss.item()
        valid_loss = valid_loss/valid_data_size
        print('\n dev loss:{:.2f}'.format(valid_loss))

end = time.time()
print('time using:{}'.format(end - begin))
torch.save(model, './filtering/models/bert-base-finetuned.pt')


        