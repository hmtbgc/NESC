python train.py --dataset flickr --n_layer 4 --partsize 5000 --batch_size 5 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 50 --every_val 5 --memory 0
python train.py --dataset ogbn-arxiv --n_layer 4 --partsize 5000 --batch_size 5 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 50 --every_val 5 --memory 0
python train.py --dataset reddit --n_layer 4 --partsize 8000 --batch_size 5 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 50 --every_val 2 --memory 0
python train.py --dataset amazon --n_layer 4 --partsize 8000 --batch_size 5 --h_feats 512 --dropout 0.1 --lr 0.005 --wd 0 --epoch 50 --every_val 2 --memory 0
python train.py --dataset ogbn-products --n_layer 4 --partsize 8000 --batch_size 5 --h_feats 512 --dropout 0.5 --lr 0.005 --wd 0 --epoch 50 --every_val 2 --memory 0
