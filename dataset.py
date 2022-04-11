import torch
import dgl
from utils import *

class IGMCDataset(torch.utils.data.Dataset):
    def __init__(self, graph, nodes, labels, num_features, h, train):
        super(IGMCDataset, self).__init__()
        self.graph = graph
        self.nodes = nodes # train_nodes
        self.labels = labels # train_ratings
        self.hop = h
        self.train = train
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        target_u_node,  target_v_node = self.nodes[idx]
        label = self.labels[idx]
        subgraph = extract_subgraph(target_u_node, target_v_node, self.graph, self.hop, self.train)

        return subgraph, label
    