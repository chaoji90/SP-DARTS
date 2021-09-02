##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import math, random, torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from ..cell_operations import OPS


CHANNEL_REDUCTION = 4
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedOp(nn.Module):
    def __init__(self, op_names, C_in, C_out, stride, affine):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()

        for primitive in op_names:
          op = OPS[primitive](C_in//CHANNEL_REDUCTION, C_out //CHANNEL_REDUCTION, stride, affine, True)
          if 'pool' in primitive:
            op = nn.Sequential(op, nn.BatchNorm2d(C_out//CHANNEL_REDUCTION, affine=False))
          self._ops.append(op)


    def forward(self, x, weights):
        #channel proportion k=4  
        dim_2 = x.shape[1]
        xtemp = x[ : , :  dim_2//CHANNEL_REDUCTION, :, :]
        xtemp2 = x[ : ,  dim_2//CHANNEL_REDUCTION:, :, :]
        self._fs = [op(xtemp) for op in self._ops]
        temp1 = sum(w * op for w, op in zip(weights, self._fs))
        #reduction cell needs pooling before concat
        if temp1.shape[2] == x.shape[2]:
          ans = torch.cat([temp1,xtemp2],dim=1)
        else:
          ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
        ans = channel_shuffle(ans, CHANNEL_REDUCTION)
        #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
        #except channe shuffle, channel shift also works
        return ans

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
            op = MixedOp(self.op_names, C_in, C_out, stride, affine)
            #xlists = [OPS[op_name](C_in , C_out, stride, affine, track_running_stats) for op_name in op_names]
        else:
            op = MixedOp(self.op_names, C_in, C_out, 1, affine)
            #xlists = [OPS[op_name](C_in , C_out,      1, affine, track_running_stats) for op_name in op_names]
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
