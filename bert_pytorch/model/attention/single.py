import torch.nn as nn
import torch.nn.functional as F
import torch
from model.attention.SoftmaxLayer import softmax_Norm
import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self):
        super(Attention, self).__init__()
        self.normbound = nn.Parameter(torch.randn([1], requires_grad=True) ** 2)
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        
        # p_attn = F.softmax(scores, dim=-1)
        p_attn = softmax_Norm(scores, dim=-1, normbound=self.normbound)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


# class Attention(nn.Module):
#     """
#     Compute 'Scaled Dot Product Attention
#     """
#     # def __init__(self):
#     #     super(Attention, self).__init__()
#     #     self.normbound = nn.Parameter(torch.randn([1], requires_grad=True) ** 2)
#     def forward(self, query, key, value, mask=None, dropout=None):
#         scores = torch.matmul(query, key.transpose(-2, -1)) \
#                  / math.sqrt(query.size(-1))

#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)

        
#         p_attn = F.softmax(scores, dim=-1)
#         # p_attn = softmax_Norm(scores, dim=-1, normbound=self.normbound)

#         if dropout is not None:
#             p_attn = dropout(p_attn)

#         return torch.matmul(p_attn, value), p_attn

