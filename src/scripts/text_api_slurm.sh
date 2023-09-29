# partition=AI4Good_X
partition=$1
   
srun -p ${partition} --gres=gpu:1 --ntasks-per-node=1 --kill-on-bad-exit \
    python text_api.py \