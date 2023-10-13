import random

def load_data(batch_size = 10, path = './filtering/rd/train.txt', shuffle = True):
    with open(path, 'r', encoding = 'UTF-8') as f:
        lines = f.readlines()
    dataset = []
    for i in range(len(lines)//batch_size):
        this_batch = []
        for line in lines[i*batch_size : (i + 1)*batch_size]:
            arg, h1,h2,h3,h4 = line.strip('\n').split('\t')
            this_batch.append({'arg':arg, 'hyps':[h1,h2,h3,h4]})
        dataset.append(this_batch)
    if shuffle:
        random.seed(0)
        random.shuffle(dataset)
    return dataset, len(lines)