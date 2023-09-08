# partition=AI4Good_X
partition=$1
   
srun -p ${partition} --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
   python inference_mc_once.py \
    --model lamm_peft \
    --vision_type image \
    --encoder_pretrain mineclip \
    --encoder_ckpt_path ../model_zoo/mineclip_ckpt/mineclip_image_encoder_vit-B_196tokens.pth \
    --llm_ckpt_path ../model_zoo/vicuna_ckpt/13b_v0/ \
    --delta_ckpt_path ../ckpt/LAMM_MineClip_36k_Vicuna_13b_v0/pytorch_model.pt \
    --vision_feature_type local \
    --vision_output_layer -2 \
    --num_vision_token 196 \
    --conv_mode vicuna_v1_1 \
    --task_type minecraft
