# Llama3.1 8B inference (generation)
## Description
This is a guide for how to run LLama3.18B in tp mode with shm allreduce backend.

### Preparation
1. Install required packages.
```bash
pip install datasets sentencepiece psutil

# Install openmp and mkl
conda install mkl
```

2. Install Pytorch
```bash
git clone https://github.com/gaopengff/pytorch.git -b gaopengf/shm_gloo
cd pytorch
git submodule sync
git submodule update --init --recursive
python setup.py install
```

3. Install transformers
```bash
git clone https://github.com/Valentine233/transformers.git -b test_flex_attn
cd transformers
python setup.py install
```

4. Download prompt.json
```bash
# Get prompt.json for gneration inference
wget -O prompt.json https://intel-extension-for-pytorch.s3.amazonaws.com/miscellaneous/llm/prompt-3.json
```

### Running inference
In mymodel folder, make sure prompt.json exists in the same folder.
```bash
cd pytorch/mymodel

1. Run inference with launcher.py
```
Running inference with tp=6.

```bash
export NUM_KV_HEADS=8

DTYPE=bf16
OUTPUT_TOKEN=1024
INPUT_TOKEN=1024
BATCH_SIZE=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
NUM_WARMUP=2
NUM_ITER=4
WORLD_SIZE=6


python launcher.py \
    --world_size $WORLD_SIZE \
    --master_addr="127.0.0.1" \
    --master_port=29500 \
    --bind_cores_to_rank \
    run_llm_inference.py \
    -m $MODEL \
    --dtype $DTYPE \
    --input-tokens ${INPUT_TOKEN} \
    --max-new-tokens ${OUTPUT_TOKEN} \
    --batch-size ${BATCH_SIZE} \
    --num-warmup ${NUM_WARMUP} \
    --num-iter ${NUM_ITER} \
    --page-size 64 \
    --profile \
    --inductor

```

2. Run inference with torchrun and xeon.launch
```bash
export NUM_KV_HEADS=8

DTYPE=bf16
OUTPUT_TOKEN=1024
INPUT_TOKEN=1024
BATCH_SIZE=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
NUM_WARMUP=2
NUM_ITER=4
WORLD_SIZE=6

torchrun --nproc-per-node=6 \
    -m torch.backends.xeon.run_cpu \
    --ncores-per-instance 40 \
    run_llm_inference.py \
    -m $MODEL \
    --dtype $DTYPE \
    --input-tokens ${INPUT_TOKEN} \
    --max-new-tokens ${OUTPUT_TOKEN} \
    --batch-size ${BATCH_SIZE} \
    --num-warmup ${NUM_WARMUP} \
    --num-iter ${NUM_ITER} \
    --page-size 64 \
    --profile \
    --inductor

```

Note: Xeon launcher requires change of https://github.com/LifengWang/pytorch/pull/4. The current branch will keep code changes alighed with this PR.