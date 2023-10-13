import os
import sys

import fire
import torch
from tqdm import tqdm
from utils.prompter import Prompter
import numpy as np



def main(
    model = 'gpt3' # bart, gpt2, dialogpt, llama, alpaca, gpt3, argllama, noargjudge, noprompt
):

    filtering_model = torch.load("./filtering/models/bert-base-finetuned.pt")
    with open('argtersely/test/argument.txt', 'r', encoding='UTF-8') as f:
        args = [line.strip() for line in f.readlines()]
    with open('experiments/{}_res.txt'.format(model), 'r', encoding='UTF-8') as f:
        con_args = [line.strip() for line in f.readlines()]
    score_list = []
    for arg, con_arg in zip(args, con_args):
        score = filtering_model.inference(arg, con_arg)
        score_list.append(score)
    x = 1 - np.mean(score_list)/4# raw_output, [0, 1]
    def calculate(x):
        if x < 0.9:
            return x/9
        else:
            return 9*x - 8        
    y = calculate(x) # expand [0.9, 1] -> [0.1, 1]
    print("arg-judge score of model {} is:{}".format(model, y))

if __name__ == "__main__":
    fire.Fire(main)