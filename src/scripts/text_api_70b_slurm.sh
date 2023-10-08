# partition=AI4Good_X
partition=$1

numgpu=4
   
srun -p ${partition} --gres=gpu:${numgpu} --ntasks-per-node 1 --kill-on-bad-exit \
torchrun --nnodes=1 --nproc_per_node=1 --master_port=25440 text_api_70b.py \