##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import math, random, torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from ..cell_operations import OPS


class MixedOp(nn.Module):
    def __init__(self, xlists):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList(xlists)

    def forward(self, x, weights):
        self._fs = [op(x) for op in self._ops]
        return sum( op * w for op, w in zip(self._fs, weights) )

def set_grad(module, input_grad, output_grad):
    module.output_grad_value = output_grad

# This module is used for NAS-Bench-201, represents a small search space with a complete DAG
class NAS201SearchCell(nn.Module):

  def __init__(self, C_in, C_out, stride, max_nodes, op_names, affine=False, track_running_stats=True):
    super(NAS201SearchCell, self).__init__()

    self.op_names  = deepcopy(op_names)
    self.edges     = nn.ModuleDict()
    self.max_nodes = max_nodes
    self.in_dim    = C_in
    self.out_dim   = C_out
    self._ops = []
    for i in range(1, max_nodes):
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        if j == 0:
          xlists = [OPS[op_name](C_in , C_out, stride, affine, track_running_stats) for op_name in op_names]
        else:
          xlists = [OPS[op_name](C_in , C_out,      1, affine, track_running_stats) for op_name in op_names]
        op = MixedOp(xlists)
        op.register_backward_hook(set_grad)
        self._ops.append(op)
        self.edges[ node_str ] = op
    self.edge_keys  = sorted(list(self.edges.keys()))
    self.edge2index = {key:i for i, key in enumerate(self.edge_keys)}
    self.num_edges  = len(self.edges)

  def extra_repr(self):
    string = 'info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}'.format(**self.__dict__)
    return string

  def forward(self, inputs, weightss):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        weights  = weightss[ self.edge2index[node_str] ]
        inter_nodes.append( self.edges[node_str](nodes[j], weights) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]

  # Forward with edge weights
  def forward_edge_weights(self, inputs, weightss, edge_weights):
    nodes = [inputs]
    for i in range(1, self.max_nodes):
      inter_nodes = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        edge_index = self.edge2index[node_str]

        weights  = weightss[ edge_index ]
        inter_nodes.append( edge_weights[edge_index] * self.edges[node_str](nodes[j], weights) )
      nodes.append( sum(inter_nodes) )
    return nodes[-1]
