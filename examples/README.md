# How to run distributed inference 

## Requirement packages
|Package|Link|
|--|--|
|Pytorch| https://github.com/gaopengff/pytorch/tree/test_pr|
|Torchccl| https://github.com/intel/torch-ccl/tree/v2.7.0%2Bcpu|
|Transformers| https://github.com/Valentine233/transformers/tree/test_flex_attn|

### 1. Set up environment
Install pytorch
```bash
git clone -b test_pr https://github.com/gaopengff/pytorch.git
cd pytorch
python setup.py install
```
Install torch-ccl
```bash
git clone https://github.com/intel/torch-ccl.git
cd pytorch
git checkout v2.7.0+cpu
git apply torchccl_fix_xpu.patch
python setup.py install
```

Install transformers
```bash
git clone https://github.com/Valentine233/transformers.git
cd transformers
git checkout test_flex_attn
python setup.py install
```

### 2. Run inference
```bash
# source oneccl env
source /home/sdp/miniforge3/envs/<your_env>/lib/python3.10/site-packages/oneccl_bind_pt-2.7.0+cpu-py3.10-linux-x86_64.egg/oneccl_bindings_for_pytorch/env/setvars.sh

# run model with numctl in bash
bash ./llama_tp.sh
``` 
