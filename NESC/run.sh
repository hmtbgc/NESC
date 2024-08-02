python train.py --dataset flickr --n_layer 3 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --s1 5 --s2 10 --layer_max 20000 

python train.py --dataset ogbn-arxiv --n_layer 3 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --s1 5 --s2 10 --layer_max 50000

python train.py --dataset reddit --n_layer 3 --h_feats 256 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --s1 5 --s2 10 --layer_max 10000

python train.py --dataset yelp --n_layer 3 --h_feats 512 --dropout 0.1 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --s1 5 --s2 10 --layer_max 20000

python train.py --dataset amazon --n_layer 3 --h_feats 512 --dropout 0.1 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --s1 5 --s2 10 --layer_max 30000

python train.py --dataset ogbn-products --n_layer 3 --h_feats 512 --dropout 0.5 --lr 0.005 --wd 0 --epoch 100 --every_val 5 --batch_size 10000 --s1 5 --s2 10 --layer_max 50000

