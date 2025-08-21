import torch
import os
import argparse
import math
import time
import pathlib
import numpy as np
import json
import sys
from itertools import chain

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from torch._inductor import config as inductor_config

parser = argparse.ArgumentParser(
    "LLM generation (greedy search) script for inductor torch.compile path",
    add_help=False,
)
parser.add_argument(
    "-m",
    "--model-name-or-path",
    default="meta-llama/Llama-2-7b-hf",
    type=str,
    help="path to model or model name in HF hub",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["fp32", "bf16", "int8-bf16", "int8", "fp16"],
    help="bf16 or fp32",
    default="bf16",
)
parser.add_argument(
    "--max-new-tokens", default=32, type=int, help="output max new tokens"
)
parser.add_argument("--input-tokens", default="32", type=str)
parser.add_argument("--page-size", default=32, type=int)
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--num-iter", default=100, type=int, help="num iter")
parser.add_argument("--num-warmup", default=10, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument(
    "--group-size", default=64, type=int, help="group size for woq int4"
)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--accuracy_only", action="store_true")
parser.add_argument("--dataset", nargs="?", default="lambada", const="lambada")
parser.add_argument("--disable-concat-linear", action="store_true")
parser.add_argument("--disable-grouped-gemm", action="store_true")
parser.add_argument(
    "--weight-dtype",
    type=str,
    choices=["INT8", "INT4"],
    help="int8 or int4",
    default="INT8",
)
# below are args for scripts compatibility, will be refined.
parser.add_argument(
    "--weight-only-quant",
    action="store_true",
    help="use weight-only quantization",
)
parser.add_argument("--torchao", action="store_true")
parser.add_argument("--lambada", action="store_true")
parser.add_argument("--inductor", action="store_true", default=False)
parser.add_argument("--enable-tp", action="store_true")
parser.add_argument("--tp-accelerate-launcher", action="store_true")
parser.add_argument("--token-latency", action="store_true")
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--int8-qconfig", nargs="?", default="./qconfig.json")
parser.add_argument(
    "--int8_bf16_mixed",
    action="store_true",
    help="by default it is int8-fp32 mixed, to enable int8 mixed amp bf16 (work on platforms like SPR)",
)
parser.add_argument("--asym-quant-act", action="store_true")



def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

def main():
    args = parser.parse_args()

    if args.dtype == "bf16":
        amp_enabled = True
        load_dtype = torch.bfloat16
    elif args.dtype == "fp32":
        amp_enabled = False
        load_dtype = torch.float
    elif args.dtype in ["int8-bf16", "int8"]:
        # mixed bf16
        amp_enabled = True
        load_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        amp_enabled = True
        load_dtype = torch.float16
    else:
        raise SystemExit(
            "This script (inductor peak perf with flexAttention) only support "
            "int8, bf16, fp16, int8-bf16, int8 (da8w8) and fp32 as dtype"
        )
    
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    enable_tp = True if world_size > 1 else False


    if enable_tp:
        import torch.distributed as dist
        rank = int(os.environ.get('RANK', 0))
        os.environ['PMI_RANK'] = str(rank)
        os.environ['PMI_SIZE'] = str(world_size)
        # 'LOCAL_WORLD_SIZE' is for shm_allreduce
        os.environ['LOCAL_WORLD_SIZE'] = str(world_size)
        dist.init_process_group("gloo")


    model_kwargs = dict(torch_dtype=load_dtype)
    if enable_tp:
        model_kwargs["tp_plan"] = "auto"
    else:
        model_kwargs["device_map"] = "cpu"

    attn_type = "paged_attention"
    model_kwargs["attn_implementation"] = attn_type

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, **model_kwargs
    )

    if attn_type == "paged_attention":
        model.generation_config.cache_implementation = "paged"
        model.config.page_size = args.page_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    tp_count = world_size
    if enable_tp:
        origin_num_kv_heads = model.config.num_key_value_heads
        model.config.num_key_value_heads = math.ceil(model.config.num_key_value_heads / tp_count)
        num_kv_heads = model.config.num_key_value_heads
        model.config.hidden_size = num_kv_heads * (model.config.hidden_size // origin_num_kv_heads)
        model.config.num_attention_heads = num_kv_heads * (model.config.num_attention_heads // origin_num_kv_heads)

    if args.inductor:
        # inductor setting
        if args.profile:
            inductor_config.profiler_mark_wrapper_call = True
            inductor_config.cpp.enable_kernel_profile = True
            inductor_config.cpp.descriptive_names = "inductor_node"
        
        inductor_config.cpp_wrapper = True
        inductor_config.max_autotune = True
        inductor_config.max_autotune_gemm_backends = "CPP,ATEN"
        torch._dynamo.config.allow_unspec_int_on_nn_module = True

        if not args.disable_grouped_gemm and hasattr(
            inductor_config.cpp, "enable_grouped_gemm_template"
        ):
            inductor_config.cpp.enable_grouped_gemm_template = True
        elif not args.disable_concat_linear:
            inductor_config.cpp.enable_concat_linear = True
        
        with torch.no_grad(), torch.autocast("cpu", enabled=amp_enabled, dtype=load_dtype):
            model.forward = torch.compile(model.forward)


    if enable_tp:
        dist.barrier()

    
    # greedy search
    generate_kwargs = dict(do_sample=False, temperature=0.9, num_beams=1)
    current_path = pathlib.Path(__file__).parent.resolve()

    model_type = "[not support model]"
    if "llama" in args.model_name_or_path.lower():
        model_type = "llama"
    elif "gpt-j" in args.model_name_or_path.lower():
        model_type = "gpt-j"

    if args.prompt is not None:
        prompt = args.prompt
    else:
        with open(str(current_path) + "/prompt.json") as f:
            prompt_pool = json.load(f)
        if model_type in prompt_pool and args.input_tokens in prompt_pool[model_type]:
            prompt = prompt_pool[model_type][args.input_tokens]
        else:
            raise SystemExit(
                "[ERROR] No such input_tokens prompt in prompt.json, Plese use --prompt if want to use custom input."
            )
        
    input_size = tokenizer(prompt, return_tensors="pt").input_ids.size(dim=1)

    prompt = [prompt] * args.batch_size
    

    
    # warmup
    with torch.no_grad(), torch.autocast("cpu", enabled=amp_enabled, dtype=load_dtype):
        for i in range(args.num_warmup):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            model.generate(input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs)
    
    # Instruct the torch.compile runtime to run minimal set of guards
    torch.compiler.set_stance(skip_guard_eval_unsafe=True)
    if args.profile:
        if args.enable_tp:
            # Wait for all ranks to finish before move on
            dist.barrier()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
            on_trace_ready=trace_handler,
            record_shapes=True,
        ) as prof:
            with torch.no_grad(), torch.autocast(
                "cpu", enabled=amp_enabled, dtype=load_dtype
            ):
                for i in range(3):
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
                    model.generate(
                        input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs
                    )
                    prof.step()
    
    # benchmark
    if args.enable_tp:
        # Wait for all ranks to finish before move on
        dist.barrier()
    num_iter = args.num_iter - args.num_warmup
    total_time = 0.0
    total_list = []
    
    with torch.no_grad(), torch.autocast("cpu", enabled=amp_enabled, dtype=load_dtype):
        for i in range(num_iter):
            tic = time.time()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

            output = model.generate(input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs)

            #output = model(input_ids, max_new_tokens=args.max_new_tokens, **generate_kwargs)

            gen_ids = output[0]
            gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            toc = time.time()
            total_time += toc - tic
            total_list.append(output[1])
    

    if args.enable_tp:
        # Wait for all ranks to finish before move on
        dist.barrier()

    print(gen_text, flush=True)
    print("\n", "-" * 10, "Summary:", "-" * 10)
    latency = total_time / (num_iter)
    print("inference-latency: %.3f sec." % latency)
    
    first_latency = np.mean([x[0] for x in total_list])
    next_latency_list = list(chain(*[x[1:] for x in total_list]))
    next_latency_list.sort()
    average_next_latency = np.mean(next_latency_list)
    p90_latency = np.percentile(next_latency_list, 90)
    print("first-token-latency: %.3f sec." % first_latency)
    print("rest-token-latency: %.3f sec." % average_next_latency)
    print("P90-rest-token-latency: %.3f sec." % p90_latency)
    


if __name__ == "__main__":
    main()