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
import math, random

def train(g, model, args, device):
    st = time.time()
    
    num_partitions = math.ceil(g.num_nodes() / args.partsize)
    
    ids = dgl.metis_partition_assignment(g, k=num_partitions)
    
    tot_part_ids = []
    for i in range(num_partitions):
        node_ids = torch.where(ids == i)[0]
        tot_part_ids.append(node_ids)
        
    random.shuffle(tot_part_ids)
    
    batch_datas = []
    
    for i in range(0, len(tot_part_ids), args.batch_size):
        sg_node_ids = tot_part_ids[i : i + args.batch_size]
        sg_node_ids = torch.cat(sg_node_ids).unique()
        sg = g.subgraph(sg_node_ids)
        batch_datas.append(sg)
        
    ed = time.time()
    
    PRINT_LOG(f'preprocess time: {ed - st:.2f}s', file=file)
    
    
    history = []
    train_time = 0
    best_val_f1 = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    for e in range(args.epoch):
        
        st = time.time()
        model.train()
        
        for sg in batch_datas:
            sg = sg.to(device) 
            x = sg.ndata['feat'].to(device)
            y = sg.ndata['label'].to(device)
            mask = sg.ndata["train_mask"].bool().to(device)
            pred = model.fullbatch_forward(sg, x)
            if (args.multilabel):
                loss = F.binary_cross_entropy_with_logits(pred[mask], y[mask].type_as(pred), pos_weight=None, reduction="mean")
            else:
                loss = F.cross_entropy(pred[mask], y[mask], reduction="mean")
    
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
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
    
    num_partitions = math.ceil(g.num_nodes() / args.partsize)
    
    ids = dgl.metis_partition_assignment(g, k=num_partitions)
    
    tot_part_ids = []
    for i in range(num_partitions):
        node_ids = torch.where(ids == i)[0]
        tot_part_ids.append(node_ids)
        
    random.shuffle(tot_part_ids)
    
    batch_datas = []
    
    for i in range(0, len(tot_part_ids), args.batch_size):
        sg_node_ids = tot_part_ids[i : i + args.batch_size]
        sg_node_ids = torch.cat(sg_node_ids).unique()
        sg = g.subgraph(sg_node_ids)
        batch_datas.append(sg)
        
    avd = [sg.in_degrees().sum() / sg.num_nodes() for sg in batch_datas]
    PRINT_LOG(f"ave_d: {math.ceil(np.mean(avd))}", file=file)
    
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    for sg in batch_datas:
        sg = sg.to(device) 
        x = sg.ndata['feat'].to(device)
        y = sg.ndata['label'].to(device)
        mask = sg.ndata["train_mask"].bool().to(device)
        pred = model.fullbatch_forward(sg, x)
        if (args.multilabel):
            loss = F.binary_cross_entropy_with_logits(pred[mask], y[mask].type_as(pred), pos_weight=None, reduction="mean")
        else:
            loss = F.cross_entropy(pred[mask], y[mask], reduction="mean")

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        opt.step()
        
        peak_usage = torch.cuda.max_memory_allocated(device)
        max_mem = max(max_mem, peak_usage)
        torch.cuda.empty_cache()
        
    PRINT_LOG(f'cuda peak usage: {max_mem / (1024 ** 2):.2f}MB', file=file)
    
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epoch", type=int, default=50, help="Maximum training epoch")
    parser.add_argument("--every_val", type=int, default=2, help="# epoch for each validation")
    parser.add_argument("--h_feats", type=int, default=256, help="Hidden feature size")
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--partsize", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--memory", type=int, default=0, help="whether check max allocated GPU memory")
    args = parser.parse_args()
    
    if os.path.exists('./cluster_gcn.pkl'):
        os.remove('./cluster_gcn.pkl')
        
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

    
        