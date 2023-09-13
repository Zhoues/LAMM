# partition=AI4Good_X

partition=$1
dataset=minecraft

exp=Mine_36k_simple_reply_Vicuna_13b_1.5

base_data_path=../datasets/2D_Benchmark
token_num=196
layer=-2
answerdir=../answers
mkdir -p ${answerdir}/${exp}
results_path=../results
mkdir -p ${results_path}/${exp}

srun -p ${partition} --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python inference_2d.py \
        --model lamm_peft \
        --encoder_pretrain mineclip \
        --llm_ckpt_path ../model_zoo/vicuna_ckpt/13b_1.5 \
        --encoder_ckpt_path ../model_zoo/mineclip_ckpt/mineclip_image_encoder_vit-B_196tokens.pth \
        --delta_ckpt_path ../ckpt/${exp}/pytorch_model.pt \
        --max_tgt_len 400 \
        --lora_r 32 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --num_vision_token ${token_num} \
        --vision_output_layer ${layer} \
        --conv_mode default \
        --dataset-name ${dataset} \
        --base-data-path ${base_data_path} \
        --inference-mode common \
        --bs 32 \
        --answers-dir ${answerdir}/${exp} \
