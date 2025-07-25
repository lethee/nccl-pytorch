export OMP_NUM_THREADS=1

curl -so- https://raw.githubusercontent.com/lethee/nccl-pytorch/refs/heads/main/nccl_perf_test.py

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NPROCS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    nccl_perf_test.py

# bash -c 'curl -so- https://raw.githubusercontent.com/lethee/nccl-pytorch/refs/heads/main/run_remote.py | bash'