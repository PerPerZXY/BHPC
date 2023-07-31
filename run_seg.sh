#!/bin/bash

#source activate py37
#dataset=mmwhs
dataset=hippo
fold=1
train_sample=0.4
method=mix
#model_path=SupCon_Hippocampus_adam_fold_2_lr_0.0001_decay_0.0001_bsz_4_temp_0.1_train_1.0_mlp_stride_pretrained
# notice: when load saved models, remember to check whether true model is loaded
python run_mix_pipeline.py  --dataset hippo --train_sample 0.2 --fold 1 --batch_size 8 \
        --load_saved_model
        #--saved_model_path  data/checkpoint_last.pth.tar\
