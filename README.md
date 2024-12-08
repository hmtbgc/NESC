## Towards Building a Lightweight and Powerful Computation Graph for Scalable GNN 
Accepted by WISE 2024 [paper link](https://link.springer.com/chapter/10.1007/978-981-96-0567-5_14)
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

If you find our work useful, please cite our paper:

```
@InProceedings{NESC, 
author={Gong, Chang and Yang, Boyu and Zheng, Weiguo and Huang, Xing and Huang, Weiyi and Wen, Jie and Sun, Shijie and Yang, Bohua}, 
title={Towards Building a Lightweight and Powerful Computation Graph for Scalable GNN}, 
booktitle={Web Information Systems Engineering -- WISE 2024}, 
year={2025}, 
pages={177--192} 
}
```
