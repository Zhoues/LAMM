#!/bin/bash
numgpu=8

# partition=AI4Good_X
# dataname=Mine_7k
partition=$1
dataname=$2
llmname=LLaMA2_13b_chat
epoch=2
exp=${dataname}_${llmname}_epoch_${epoch}
visfeat_type=local

now=$(date +"%Y%m%d_%H%M%S")
ckpt_dir=../ckpt
mkdir -p ${ckpt_dir}/${exp}/log_rest/

srun -p ${partition} -J ${exp} --gres=gpu:${numgpu} --ntasks-per-node 1 --kill-on-bad-exit \
torchrun --nnodes=1 --nproc_per_node=${numgpu} --master_port=25440 train.py \
    --stage 1 \
    --cfg ./config/train.yaml \
    --data_path  ../datasets/2D_Instruct/${dataname}/${dataname}_instruct.json \
    --vision_root_path ../datasets/2D_Instruct/${dataname}/${dataname}_image/ \
    --conv_template llama2 \
    --max_tgt_len 400 \
    --vision_type image \
    --use_system \
    --model lamm_peft \
    --encoder_pretrain mineclip \
    --encoder_ckpt_path ../model_zoo/mineclip_ckpt/mineclip_image_encoder_vit-B_196tokens.pth \
    --llm_ckpt_path ../model_zoo/llama2_ckpt/13b_chat/ \
    --vision_feature_type ${visfeat_type} \
    --num_vision_token 196 \
    --save_path  ${ckpt_dir}/${exp} \
    --log_path ${ckpt_dir}/${exp}/log_rest/ \
    2>&1 | tee ${ckpt_dir}/${exp}/log_rest/train_${now}.log

