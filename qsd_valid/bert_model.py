from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

class RM_model(nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, 1).to(self.device)
        self.softmax = nn.Softmax(dim = 0).to(self.device)

    def forward(self, arg, hyps:List[str], rank = torch.tensor([1, 2, 3, 4], dtype = torch.float)):
        output_rank = []
        for hyp in hyps:
            encoded_input = self.tokenizer(arg + '[SEP]' + hyp, return_tensors = 'pt').to(self.device)
            output = self.bert(**encoded_input)
            linear_output = self.linear(output.pooler_output)
            output_rank.append(linear_output.view(-1))
        output_rank_tensor = torch.cat(output_rank, dim = 0).to(self.device)
        output_rank_tensor_sfmax = self.softmax(output_rank_tensor)
        target_rank_tensor_sfmax = self.softmax(rank).to(self.device)
        kl_loss = F.kl_div(output_rank_tensor_sfmax.log(), target_rank_tensor_sfmax, reduction='sum')
        return kl_loss
    
    def inference(self, arg, hyps:List[str]):
        output_rank = []
        for hyp in hyps:
            encoded_input = self.tokenizer(arg + '[SEP]' + hyp, return_tensors = 'pt').to(self.device)
            output = self.bert(**encoded_input)
            linear_output = self.linear(output.pooler_output)
            output_rank.append(linear_output.item())
        #return hyps[np.argmin(output_rank)]
        return np.argmin(output_rank)




