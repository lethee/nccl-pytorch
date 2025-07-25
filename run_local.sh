export OMP_NUM_THREADS=1

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NPROCS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    /host/nccl_perf_test.py
