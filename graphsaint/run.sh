python train.py --dataset "flickr" --num_roots 2048 --lr 0.005 --wd 0.0 --hid 256 --epoch 50 --every_val 2 --n_layer 4 --dropout 0.5 --memory 0
python train.py --dataset "ogbn-arxiv" --num_roots 2048 --lr 0.005 --wd 0.0 --hid 256 --epoch 50 --every_val 2 --n_layer 4 --dropout 0.5 --memory 0
python train.py --dataset "reddit" --num_roots 2048 --lr 0.005 --wd 0.0 --hid 256 --epoch 50 --every_val 2 --n_layer 4 --dropout 0.5 --memory 0
python train.py --dataset "amazon" --num_roots 2048 --lr 0.005 --wd 0.0 --hid 512 --epoch 50 --every_val 2 --n_layer 4 --dropout 0.1 --memory 0
python train.py --dataset "ogbn-products" --num_roots 2048 --lr 0.005 --wd 0.0 --hid 512 --epoch 50 --every_val 2 --n_layer 4 --dropout 0.5 --memory 0
