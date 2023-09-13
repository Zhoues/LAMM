# partition=AI4Good_X
partition=$1
   
srun -p ${partition} --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python common_eval_2d.py \
    --dataset-name minecraft \
    --answer-file ../answers/LAMM_MineClip_36k_simple_reply_Vicuna_13b_v0/minecraft_minecraft.json\
    --base-data-path ../datasets/2D_Benchmark \