python train.py --dataset "flickr" --num_roots 3000 --lr 0.005 --wd 0.0 --hid 256 --epoch 100 --every_val 5 --n_layer 3 --length 2 --num_repeat 50 --dropout 0.5
python train.py --dataset "ogbn-arxiv" --num_roots 3000 --lr 0.005 --wd 0.0 --hid 256 --epoch 100 --every_val 5 --n_layer 3 --length 2 --num_repeat 50 --dropout 0.5 
python train.py --dataset "reddit" --num_roots 3000 --lr 0.005 --wd 0.0 --hid 256 --epoch 100 --every_val 5 --n_layer 3 --length 2 --num_repeat 50 --dropout 0.5
python train.py --dataset "yelp" --num_roots 3000 --lr 0.005 --wd 0.0 --hid 512 --epoch 100 --every_val 5 --n_layer 3 --length 2 --num_repeat 50 --dropout 0.1 
python train.py --dataset "amazon" --num_roots 3000 --lr 0.005 --wd 0.0 --hid 512 --epoch 100 --every_val 5 --n_layer 3 --length 2 --num_repeat 50 --dropout 0.1 
python train.py --dataset "ogbn-products" --num_roots 3000 --lr 0.005 --wd 0.0 --hid 512 --epoch 100 --every_val 5 --n_layer 3 --length 2 --num_repeat 50 --dropout 0.5 
