#!/bin/bash


OMP_NUM_THREADS=56 \
numactl -C 0-55 -m 0 torchrun \
    --nnodes=2 --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port=29500 \
    --nproc-per-node 1 llama_tp.py & \
OMP_NUM_THREADS=56 \
numactl -C 56-111 -m 1 torchrun \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="127.0.0.1" \
    --master_port=29500 \
    --nproc-per-node 1 llama_tp.py & wait