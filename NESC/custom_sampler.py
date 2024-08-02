from dgl.dataloading.base import BlockSampler
from dgl.transforms import to_block
import dgl, torch

def edge_func(edges):
    return {'d': edges.src['d']}
    
class CustomNeigborSampler(BlockSampler):
    def __init__(self,
                s1,
                s2,
                layer_max,
                n_layer,
                edge_dir="in",
                prob=None,
                mask=None,
                replace=False,
                prefetch_node_feats=None,
                prefetch_labels=None,
                prefetch_edge_feats=None,
                output_device=None):
        super().__init__(prefetch_node_feats=prefetch_node_feats,
                        prefetch_labels=prefetch_labels,
                        prefetch_edge_feats=prefetch_edge_feats,
                        output_device=output_device)
        self.s1 = s1
        self.s2 = s2
        self.layer_max = layer_max
        self.n_layer = n_layer
        self.edge_dir = edge_dir
        if mask is not None and prob is not None:
            raise ValueError(
                "Mask and probability arguments are mutually exclusive. "
                "Consider multiplying the probability with the mask "
                "to achieve the same goal."
            )
        self.prob = prob or mask
        self.replace = replace
        
    def sample_blocks(self, g, seeds, exclude_eids=None):
        blocks = []
        output_nodes = seeds
        for i in range(self.n_layer):
            #print(f'layer {i}, seeds == {seeds}')
            frontier = dgl.sampling.sample_neighbors(g, seeds, self.s1, prob="d")
            edges = frontier.edges()
            #print(f'layer {i}, edges == {edges}')
            temp_node = edges[0]
            if temp_node.shape[0] > self.layer_max:
                
                degrees = (g.in_degrees() + g.out_degrees())[temp_node]
                normalized_degrees = degrees / torch.sum(degrees)
                sampled_num = self.layer_max
                sampled_idx = torch.multinomial(normalized_degrees, sampled_num, replacement=False)
                sampled_node = temp_node[sampled_idx]
            
            else:
                sampled_node = temp_node
                
            #print(f'layer {i}, sampled_node: {sampled_node}')
                
            node = torch.unique(torch.concat([sampled_node, seeds]))
            sg = dgl.node_subgraph(g, node)
            #sg = sg.add_self_loop()
            m = sg.ndata[dgl.NID]
            new_u = m[sg.edges()[0]]
            new_v = m[sg.edges()[1]]
            new_graph = dgl.graph((new_u, new_v), num_nodes=g.num_nodes())
            #d = new_graph.in_degrees()
            # if torch.any(d == 0):
            #     print(f'layer {i}, new_graph contains 0 in_degree')
            
            # if torch.any(seeds >= new_graph.num_nodes()):
            #     print(f"layer {i}, wtf")
            #new_graph.ndata['d'] = (new_graph.in_degrees() + new_graph.out_degrees()).float()
            #new_graph.apply_edges(edge_func)
            # new_frontier = dgl.sampling.sample_neighbors(new_graph, seeds, self.s2, prob="d")
            new_frontier = dgl.sampling.sample_neighbors(new_graph, seeds, self.s2)
            #print(f'layer {i}, new_froniter {new_frontier}')
            block = to_block(new_frontier, seeds)
            blocks.insert(0, block)
            seeds = block.srcdata[dgl.NID]
        return seeds, output_nodes, blocks
        