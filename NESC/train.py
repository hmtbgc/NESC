from dataset import CustomDataset
from models import *
import argparse
import torch
import time
from sklearn.metrics import f1_score
from utils import *
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from dgl import to_block
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from logger import *


def train(g, model, args, device):
    train_idx = torch.where(g.ndata['train_mask'] == 1)[0]
    
    st = time.time()
    
    idx = torch.randperm(train_idx.shape[0])
    tot_blocks = []
    tot_seeds = []
    
    
    for i in tqdm(range(0, train_idx.shape[0], args.batch_size)):
        
        batch_idx = idx[i : i+args.batch_size]
        seeds = train_idx[batch_idx]
        blocks = []
        
        for _ in range(args.n_layer):
            frontier = dgl.sampling.sample_neighbors(g, seeds, args.s1)
            edges = frontier.edges()
            node = torch.unique(torch.cat((edges[0], seeds), dim=0))
            sg = dgl.node_subgraph(g, node)
            sg = sg.to(device)
            seeds_gpu = seeds.to(device)
            m = sg.ndata[dgl.NID]
            new_u = m[sg.edges()[0]]
            new_v = m[sg.edges()[1]]
            new_graph = dgl.graph((new_u, new_v))
            new_frontier = dgl.sampling.sample_neighbors(new_graph, seeds_gpu, args.s2)
            block = to_block(new_frontier, seeds_gpu)
            block = block.to('cpu')
            blocks.append(block)
            seeds = block.srcdata[dgl.NID]
    
        tot_blocks.append(blocks[::-1])
        tot_seeds.append(train_idx[batch_idx])
        

    ed = time.time()
    print(f'preprocess time: {ed - st:.2f}s', file=file)
    
    
    history = []
    train_time = 0
    best_val_f1 = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    for e in range(args.epoch):
        
        st = time.time()
        model.train()
        
        for i in range(len(tot_blocks)):
            blocks, seeds = tot_blocks[i], tot_seeds[i]
            batch_inputs = g.ndata['feat'][blocks[0].srcdata[dgl.NID]].to(device)
            batch_labels = g.ndata['label'][seeds].to(device)
            blocks = [blk.to(device) for blk in blocks]
            pred = model.minibatch_forward(blocks, batch_inputs)
            if (args.multilabel):
                loss = F.binary_cross_entropy_with_logits(pred, batch_labels.type_as(pred), pos_weight=None, reduction="mean")
            else:
                loss = F.cross_entropy(pred, batch_labels, reduction="mean")
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                        
        ed = time.time()
        print(f'epoch time == {ed - st:.3f}s')
        train_time += (ed - st)
        
    
        if e > 0 and e % args.every_val == 0:
            model.eval()
            val_f1 = evaluate(model, g, g.ndata['label'], g.ndata['val_mask'], args.multilabel, False)
            history.append((e, round(train_time, 2), round(val_f1, 4)))
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), os.path.join('./pt', f'{args.dataset}.pt'))
            PRINT_LOG(f'epoch {e}, val f1 {val_f1 * 100:.2f}%, best val f1 {best_val_f1 * 100:.2f}%', file=file)
            
    return train_time, history
    
    
def test(g, model, args):
    model.load_state_dict(torch.load(os.path.join('./pt', f'{args.dataset}.pt')))
    labels = g.ndata['label']
    test_mask = g.ndata['test_mask']
    test_f1 = evaluate(model, g, labels, test_mask, multilabel=args.multilabel, test=True)
    return test_f1

def mem_bench(device, g, model, multilabel, args):
    max_mem = 0
    torch.cuda.empty_cache()
    
    train_idx = torch.where(g.ndata['train_mask'] == 1)[0]
    
    idx = torch.randperm(train_idx.shape[0])
    tot_blocks = []
    tot_seeds = []
    
    
    for i in tqdm(range(0, train_idx.shape[0], args.batch_size)):
        
        batch_idx = idx[i : i+args.batch_size]
        seeds = train_idx[batch_idx]
        blocks = []
        
        for _ in range(args.n_layer):
            frontier = dgl.sampling.sample_neighbors(g, seeds, args.s1)
            edges = frontier.edges()
            node = torch.unique(torch.cat((edges[0], seeds), dim=0))
            sg = dgl.node_subgraph(g, node)
            sg = sg.to(device)
            seeds_gpu = seeds.to(device)
            m = sg.ndata[dgl.NID]
            new_u = m[sg.edges()[0]]
            new_v = m[sg.edges()[1]]
            new_graph = dgl.graph((new_u, new_v))
            new_frontier = dgl.sampling.sample_neighbors(new_graph, seeds_gpu, args.s2)
            block = to_block(new_frontier, seeds_gpu)
            block = block.to('cpu')
            blocks.append(block)
            seeds = block.srcdata[dgl.NID]
    
        tot_blocks.append(blocks[::-1])
        tot_seeds.append(train_idx[batch_idx])
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    for i in range(len(tot_blocks)):
        blocks, seeds = tot_blocks[i], tot_seeds[i]
        batch_inputs = g.ndata['feat'][blocks[0].srcdata[dgl.NID]].to(device)
        batch_labels = g.ndata['label'][seeds].to(device)
        blocks = [blk.to(device) for blk in blocks]
        pred = model.minibatch_forward(blocks, batch_inputs)
        if (args.multilabel):
            loss = F.binary_cross_entropy_with_logits(pred, batch_labels.type_as(pred), pos_weight=None, reduction="mean")
        else:
            loss = F.cross_entropy(pred, batch_labels, reduction="mean")

        opt.zero_grad()
        loss.backward()
        opt.step()
                        
        
        peak_usage = torch.cuda.max_memory_allocated(device)
        max_mem = max(max_mem, peak_usage)
        torch.cuda.empty_cache()
        
    PRINT_LOG(f'cuda peak usage: {max_mem / (1024 ** 2):.2f}MB', file=file)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epoch", type=int, default=50, help="Maximum training epoch")
    parser.add_argument("--every_val", type=int, default=2, help="# epoch for each validation")
    parser.add_argument("--h_feats", type=int, default=128, help="Hidden feature size")
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--s1", type=int, default=1, help="fanout s1")
    parser.add_argument("--s2", type=int, default=5, help="fanout s2")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout ratio")
    parser.add_argument("--memory", type=int, default=0)
    args = parser.parse_args()
    
    file = new_log(f'./log', args)    
    fullbatch_eval_dataset = set(['flickr', 'ogbn-arxiv', 'reddit'])
    minibatch_eval_dataset = set(['amazon', 'ogbn-products'])
    multilabel_dataset = set(['amazon'])
    args.multilabel = args.dataset in multilabel_dataset
    args.fullbatch = args.dataset in fullbatch_eval_dataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_root = "../dataset"
    if args.dataset == "ogbn-arxiv" or args.dataset == "ogbn-products":
        dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset, root=dataset_root))
    else:
        dataset = CustomDataset(dataset_root, args.dataset)
    g = dataset[0]
    g = dgl.to_bidirected(g, copy_ndata=True)

    in_feats = g.ndata['feat'].shape[1]
    num_classes = dataset.num_classes
    
    tot_test_f1, tot_epoch_train_time = [], []
    
    if args.memory == 1:
        model = GNN(in_feats=in_feats, h_feats=args.h_feats, num_classes=num_classes, n_layer=args.n_layer, dropout=args.dropout, device=device)
        model = model.to(device)
        mem_bench(device, g, model, args.multilabel, args)
        
    else:
        
    
        for i in range(5):
        
            model = GNN(in_feats=in_feats, h_feats=args.h_feats, num_classes=num_classes, n_layer=args.n_layer, dropout=args.dropout, device=device)
            model = model.to(device)


            train_time, history = train(g, model, args, device)
            
            PRINT_LOG(f'train time: {train_time:.2f}s', file=file)
            test_f1 = test(g, model, args)
            PRINT_LOG(f'test f1: {test_f1 * 100:.2f}%', file=file)
            
            h_time, h_val = [], []
            for temp in history:
                h_time.append(temp[1])
                h_val.append(temp[2])
            print(f'time: {h_time}', file=file)
            print(f'val: {h_val}', file=file)
            
            tot_test_f1.append(test_f1)
            tot_epoch_train_time.append(train_time / args.epoch)
            
        PRINT_LOG(f'test f1: {np.mean(tot_test_f1) * 100:.2f} ± {np.std(tot_test_f1) * 100:.2f}', file=file)
        PRINT_LOG(f'epoch time: {np.mean(tot_epoch_train_time):.2f} ± {np.std(tot_epoch_train_time):.2f}s', file=file)
        

        
        
        
        
        

    

        
    
    
    