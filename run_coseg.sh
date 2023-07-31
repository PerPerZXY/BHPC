#!/bin/bash


source activate py37

batch=4
dataset=Hippocampus
#dataset=mmwhs
fold=3
head=mlp
mode=block
temp=0.1

train_sample=1

python main_coseg.py  --batch_size 4 --dataset Hippocampus \
    --data_folder ./data \
    --learning_rate 0.0001 \
    --epochs 70 \
    --head mlp \
    --mode block\
    --fold 3 \
    --save_freq 1 \
    --print_freq 10 \
    --temp 0.1 \
    --train_sample 1 \
    --pretrained_model_path  save/simclr/Hippocampus/b_120_model.pth\
   # --pretrained_model_path save/simclr/Hippocampus/b_80_model.pth \
    # --pretrained_model_path save/simclr/Hippocampus/b_80_model.pth \

