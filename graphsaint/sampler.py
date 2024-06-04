import math
import os
import time
import torch
import random
import numpy as np
import dgl.function as fn
import dgl
from dgl.sampling import random_walk, pack_traces
from tqdm import tqdm

class SAINTSampler(object):
    def __init__(self, g, num_roots, length, args):
        self.g = g
        self.subgraphs = []
        self.N = 0
        self.length = length
        sampled_nodes = 0
        train_idx = torch.where(g.ndata['train_mask'] == 1)[0]
        idx = torch.randperm(train_idx.shape[0])
        
        for i in tqdm(range(0, train_idx.shape[0], num_roots)):
            batch_idx = idx[i : i + num_roots]
            seeds = train_idx[batch_idx]
            traces, types = random_walk(self.g, nodes=seeds, length=self.length)
            sampled_nodes, _, _, _ = pack_traces(traces, types)
            sampled_nodes = sampled_nodes.unique()
            self.subgraphs.append(self.g.subgraph(sampled_nodes))
            
        self.num_batch = len(self.subgraphs)
    
    def __len__(self):
        return self.num_batch
    
    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n < self.num_batch:
            result = self.subgraphs[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration()
    
    
    