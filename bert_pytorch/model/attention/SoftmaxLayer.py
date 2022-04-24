# from typing import Optional

from torch import Tensor
# from torch_scatter import scatter, segment_csr, gather_csr
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
# from .num_nodes import maybe_num_nodes

# class softmax_Norm(torch.nn.Module):
#     def __init__(self):
#         super(softmax_Norm, self).__init__()
#         self.normbound = nn.Parameter(torch.randn([1], requires_grad=True) ** 2)
#         self.eps = 1e-5
#         # self.normbound = nn.Parameter(torch.tensor(0.8, requires_grad=True) ** 2)
#     def forward(self, src: Tensor, index: Optional[Tensor], ptr: Optional[Tensor] = None,
#                 num_nodes: Optional[int] = None) -> Tensor:
#         r"""Computes a sparsely evaluated softmax.
#         Given a value tensor :attr:`src`, this function first groups the values
#         along the first dimension based on the indices specified in :attr:`index`,
#         and then proceeds to compute the softmax individually for each group.

#         Args:
#             src (Tensor): The source tensor.
#             index (LongTensor): The indices of elements for applying the softmax.
#             ptr (LongTensor, optional): If given, computes the softmax based on
#                 sorted inputs in CSR representation. (default: :obj:`None`)
#             num_nodes (int, optional): The number of nodes, *i.e.*
#                 :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

#         :rtype: :class:`Tensor`
#         """
#         if ptr is not None:
#             src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
#             out = (src - src_max)

#             softmaxrate = torch.sqrt(self.normbound ** 2 / (torch.var(out) + self.eps))
#             out = (out * softmaxrate).exp()
#             out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
#         elif index is not None:
#             N = maybe_num_nodes(index, num_nodes) # 计算batch_size的大小
#             """
#             取每张图中节点特征的最大值，相当于对图中的所有节点进行了一次max_pooling
#             """
#             src_max = scatter(src, index, dim=0, dim_size=N, reduce='max')[index]
#             out = (src - src_max)

#             softmaxrate = torch.sqrt(self.normbound ** 2 / (torch.var(out) + self.eps))
#             out = (out * softmaxrate).exp()
#             """
#             计算图中每个节点的注意力值，这里的注意力值并不是单纯地计算每个节点的注意力值，
#             而是每个节点中特征的注意力值。这种做法更加精细。最后图表示向量中特征是图中所有
#             节点相应特征的加权和。
#             """
#             out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
#         else:
#             raise NotImplementedError

#         return out / (out_sum + 1e-16)


# class softmax_Norm(torch.nn.Module):
#     def __init__(self):
#         super(softmax_Norm, self).__init__()
#         self.normbound = nn.Parameter(torch.randn([1], requires_grad=True) ** 2)
#         self.sigmoid_a = nn.Parameter(torch.randn([1], requires_grad=True) ** 2)
#         self.sigmoid_b = nn.Parameter(torch.randn([1], requires_grad=True) ** 2)
#         # self.normbound = nn.Parameter(torch.tensor(0.8, requires_grad=True) ** 2)
#     def forward(self, src: Tensor, index: Optional[Tensor], ptr: Optional[Tensor] = None,
#                 num_nodes: Optional[int] = None) -> Tensor:
#         r"""Computes a sparsely evaluated softmax.
#         Given a value tensor :attr:`src`, this function first groups the values
#         along the first dimension based on the indices specified in :attr:`index`,
#         and then proceeds to compute the softmax individually for each group.
#
#         Args:
#             src (Tensor): The source tensor.
#             index (LongTensor): The indices of elements for applying the softmax.
#             ptr (LongTensor, optional): If given, computes the softmax based on
#                 sorted inputs in CSR representation. (default: :obj:`None`)
#             num_nodes (int, optional): The number of nodes, *i.e.*
#                 :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
#
#         :rtype: :class:`Tensor`
#         """
#         if ptr is not None:
#             src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
#             out = (src - src_max)
#             ratescale = torch.var(out) + self.sigmoid_b ** 2 -(self. sigmoid_b ** 2) * \
#                         torch.sigmoid(torch.var(out) - self.sigmoid_a ** 2)
#             softmaxrate = torch.sqrt(self.normbound ** 2 / ratescale)
#             out = (out * softmaxrate).exp()
#             out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
#         elif index is not None:
#             N = maybe_num_nodes(index, num_nodes) # 计算batch_size的大小
#             """
#             取每张图中节点特征的最大值，相当于对图中的所有节点进行了一次max_pooling
#             """
#             src_max = scatter(src, index, dim=0, dim_size=N, reduce='max')[index]
#             out = (src - src_max)
#             ratescale = torch.var(out) + self.sigmoid_b ** 2 -(self. sigmoid_b ** 2) * \
#                         torch.sigmoid(torch.var(out) - self.sigmoid_a ** 2)
#             softmaxrate = torch.sqrt(self.normbound ** 2 / ratescale)
#
#             out = (out * softmaxrate).exp()
#             """
#             计算图中每个节点的注意力值，这里的注意力值并不是单纯地计算每个节点的注意力值，
#             而是每个节点中特征的注意力值。这种做法更加精细。最后图表示向量中特征是图中所有
#             节点相应特征的加权和。
#             """
#             out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
#         else:
#             raise NotImplementedError
#
#         return out / (out_sum + 1e-16)


def softmax_Norm(input, dim, normbound):
    # print(input.size())
    if dim == -1:
        dim = len(input.size()) - 1
    eps = 1e-5
    softmaxrate = torch.var(input.detach().clone(), dim=dim).unsqueeze(dim).repeat(1, 1, 1, input.size(3)) + eps
    softmaxrate = torch.pow(normbound ** 2 * (1.0 / softmaxrate), 0.5)
    input_N = torch.mul(softmaxrate, input)
    return F.softmax(input_N, dim = dim)