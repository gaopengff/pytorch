# 2TP: OMP_NUM_THREADS=56 numactl -C 0-55 -m 0 torchrun --nnodes=2 --node_rank=0 --master_addr="127.0.0.1" --master_port=29500 --nproc-per-node 1 tp_hf.py & OMP_NUM_THREADS=56 numactl -C 56-111 -m 1 torchrun --nnodes=2 --node_rank=1 --master_addr="127.0.0.1" --master_port=29500 --nproc-per-node 1 tp_hf.py & wait

# no TP: OMP_NUM_THREADS=56 numactl -C 0-55 -m 0 python tp_hf.py

import os
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

import time
import torch
import torch.profiler
import oneccl_bindings_for_pytorch
from pyinstrument import Profiler

import torch
import os
import torch.backends.mkldnn
import torch.backends.openmp

print(f"Using {torch.get_num_threads()} threads (PyTorch)")
print(f"OMP_NUM_THREADS={os.getenv('OMP_NUM_THREADS')}")

# # Ensure PyTorch respects the OMP setting
# torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "56")))

# print(f"Now using {torch.get_num_threads()} threads after setting manually")

model_id = "meta-llama/Llama-3.1-8B-Instruct"
#model_id = "/localdisk/huggingface/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/"

def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

def main(is_tp, rank, world_size) -> None:
    # backend = "gloo"
    # backend = "ccl" # source ~/frameworks.ai.pytorch.torch-ccl/oneccl_bindings_for_pytorch/env/setvars.sh
    print("is_tp, rank, world_size: ", is_tp, rank, world_size)
    # if is_tp:
    #     dist.init_process_group(backend)

    model_kwargs = dict(torch_dtype=torch.bfloat16)
    if is_tp:
        model_kwargs["tp_plan"] = "auto"
    else:
        model_kwargs["device_map"] = "cpu"

    # attn_type = "eager"
    attn_type = "paged_attention"
    model_kwargs["attn_implementation"] = attn_type

    # Retrieve tensor parallel model
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    print(model.dtype)
    # print(f"torch.max_memory_allocated() is {torch.max_memory_allocated()}")
    print("="*200)
    if dist.is_initialized():
        print("Backend:", dist.get_backend())
    else:
        print("Distributed process group is not initialized.")

    # Prepare input tokens
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I have not seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate,"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024).input_ids.to(model.device)
    print(f"input shape is {inputs.shape}")
    print(f"input dtype is {inputs.dtype}")


    if attn_type == "paged_attention":
        model.generation_config.cache_implementation = "paged"
        model.config.page_size = 64
    else:
        model.generation_config.cache_implementation = "static"
    if is_tp:
        model.config.hidden_size = model.config.hidden_size // 2
        model.config.num_key_value_heads = model.config.num_key_value_heads // 2
        model.config.num_attention_heads = model.config.num_attention_heads // 2

    # for i in range(1):
    #     with torch.no_grad():
    #         start = time.time()
    #         outputs = model.generate(inputs, do_sample=False, max_new_tokens=1024, min_new_tokens=1024)
    #         end = time.time()
    #         print(f"time cost {(end-start)*1000} ms")
    #         gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True,)
    #         print("Eager: ", gen_text)

    model.forward = torch.compile(model.forward)

    # warm-up to reduce time of dynamo cache look-up
    for i in range(3):
        _ = model.generate(inputs, do_sample=False, max_new_tokens=1024, min_new_tokens=1024)

    torch.compiler.set_stance(skip_guard_eval_unsafe=True)

    if is_tp:
        dist.barrier()
    
    total_time = 0.0
    cycles = 5
    warmup = 3
    # latency
    for i in range(cycles):
        t0 = time.time()
        # outputs = model.generate(inputs, do_sample=False, max_new_tokens=1, min_new_tokens=1)
        outputs = model.generate(inputs, do_sample=False, max_new_tokens=1024, min_new_tokens=1024)
        t1 = time.time()
        print("Iteration: %d, Time: %.6f sec" % (i, t1 - t0))
        if i == 0:
            gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True,)
            print(gen_text)
        if i >= warmup:
            total_time += t1 - t0

    # Wait for all ranks to finish before move on
    if is_tp:
        dist.barrier()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=3, active=1),
        on_trace_ready=trace_handler,
    ) as prof:
        for i in range(5):
            # outputs = model.generate(inputs, do_sample=False, max_new_tokens=1, min_new_tokens=1)
            outputs = model.generate(inputs, do_sample=False, max_new_tokens=1024, min_new_tokens=1024)
            prof.step()
    # Wait for all ranks to finish before move on
    if is_tp:
        dist.barrier()

    latency = total_time / (cycles - warmup) * 1000
    print("\n", "-" * 10, "Summary:", "-" * 10)
    print("Inference latency: %.2f ms." % latency)

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = str(os.environ.get('MASTER_ADDR', '127.0.0.1'))
    os.environ['MASTER_PORT'] = '29500'
    '''
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    '''
    is_tp = "RANK" in os.environ

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    os.environ['PMI_SIZE'] = str(world_size)
    os.environ['PMI_RANK'] = str(rank)

    main(is_tp, rank, world_size)
    # main(False, 0, 1) # no tp