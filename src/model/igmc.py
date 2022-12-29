from torch.optim import Adam
from torch.nn import init
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn

class IGMC(nn.Module):
    def __init__(self, in_feats, gconv=dglnn.RelGraphConv, latent_dim=[32, 32, 32, 32], 
                 num_relations=5, num_bases=2, regression=False, dropout_rate=0.2, 
                 force_undirected=False, side_features=False, n_side_features=0, 
                 multiply_by=1):
        super(IGMC, self).__init__()
        
        self.multiply_by = multiply_by
        self.regression = True
        self.dropout_rate = dropout_rate
        self.dropout = dgl.DropEdge(p=self.dropout_rate)
        self.convs = torch.nn.ModuleList()
        self.dimensions = [in_feats] + latent_dim
        
        
        self.convs.append(gconv(self.dimensions[0], self.dimensions[1], num_rels=num_relations, num_bases = num_bases, regularizer='basis'))
        for i in range(1, len(self.dimensions)-1) :
            self.convs.append(gconv(self.dimensions[i], self.dimensions[i+1], num_rels=num_relations, num_bases = num_bases, regularizer='basis'))
        
        self.lin1 = nn.Linear(2*sum(latent_dim), 128)
        self.lin2 = nn.Linear(128, 1)
        
        
    def reset_parameters(self):
        for conv in self.convs :
            init.xavier_uniform_(conv.weight)
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        
    def forward(self, graph):
        concat_states = []
        x = graph.ndata['feature'].squeeze()
        user_idx = x[:, 0] != 0
        item_idx = x[:, 1] != 0
        
        if self.training and self.dropout_rate > 0 :
            graph = self.dropout(graph)
        
        for conv in self.convs: 
            x = torch.tanh(conv(graph, x, graph.edata['r'].squeeze()))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)
        
        ## Pooling Layer
        x = torch.cat([concat_states[user_idx], concat_states[item_idx]], 1)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            return F.log_softmax(x, dim=-1)
        
    def __repr__(self):
        return self.__class__.__name__