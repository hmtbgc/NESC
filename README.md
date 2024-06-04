### How to Run
Download dataset from [Google Drive link](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) or [BaiduYun link(code:f1ao)](https://pan.baidu.com/share/init?surl=SOb0SiSAXavwAcNqkttwcg) and put it at correct place:

```
/home/user/tot_code
│   
└───graphsage
|   
└───graphsaint
| 
└───dataset/
|   |   flickr/
|   |   ogbn-arxiv/
|   |   ...
```

Only four files are needed: adj_full.npz, class_map.json, feats.npy and role.json. These public datasets are collected by GraphSAINT and are irrelevant to this paper.

If you want to run certain algorithm(e.g. NESC):
```shell
cd NESC
bash run.sh
```

Results will be saved at log directory.