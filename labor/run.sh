python train.py --dataset flickr --n_layer 4 --fan 2 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 50 --every_val 2 --batch_size 2048 --memory 0
python train.py --dataset ogbn-arxiv --n_layer 4 --fan 2 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 50 --every_val 2 --batch_size 2048 --memory 0
python train.py --dataset reddit --n_layer 4 --fan 2 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 50 --every_val 2 --batch_size 2048 --memory 0
python train.py --dataset amazon --n_layer 4 --fan 2 --h_feats 512 --dropout 0.1 --lr 0.005 --wd 0 --epoch 50 --every_val 2 --batch_size 2048 --memory 0
python train.py --dataset ogbn-products --n_layer 4 --fan 2 --h_feats 512 --dropout 0.5 --lr 0.005 --wd 0 --epoch 50 --every_val 2 --batch_size 2048 --memory 0





