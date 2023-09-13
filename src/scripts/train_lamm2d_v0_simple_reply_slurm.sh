#!/bin/bash
numgpu=8

# partition=AI4Good_X
# dataname=Mine_36k
partition=$1
dataname=$2
llmname=Vicuna_13b_v0
exp=${dataname}_simple_reply_${llmname}_ViT-L
visfeat_type=local

now=$(date +"%Y%m%d_%H%M%S")
ckpt_dir=../ckpt
mkdir -p ${ckpt_dir}/${exp}/log_rest/

srun -p ${partition} -J ${exp} --gres=gpu:${numgpu} --ntasks-per-node 1 --kill-on-bad-exit \
torchrun --nnodes=1 --nproc_per_node=${numgpu} --master_port=25440 train.py \
    --stage 1 \
    --cfg ./config/train.yaml \
    --data_path  ../datasets/2D_Instruct/${dataname}/${dataname}_instruct_simple_reply.json \
    --vision_root_path ../datasets/2D_Instruct/${dataname}/${dataname}_image/ \
    --conv_template default \
    --max_tgt_len 400 \
    --vision_type image \
    --use_system \
    --model lamm_peft \
    --encoder_pretrain clip \
    --llm_ckpt_path ../model_zoo/vicuna_ckpt/13b_v0/ \
    --vision_feature_type ${visfeat_type} \
    --num_vision_token 256 \
    --save_path  ${ckpt_dir}/${exp} \
    --log_path ${ckpt_dir}/${exp}/log_rest/ \
    2>&1 | tee ${ckpt_dir}/${exp}/log_rest/train_${now}.log

