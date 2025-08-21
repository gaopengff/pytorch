
import subprocess
from argparse import ArgumentParser, REMAINDER
import os
import psutil
import sys
import signal
import time

parser = ArgumentParser()
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument("--master_addr", default="127.0.0.1", type=str)
parser.add_argument("--master_port", default=29500, type=int)
parser.add_argument("--bind_cores_to_rank", action="store_true")

parser.add_argument("--bind_core_list", type=str, default=None)
parser.add_argument("script", type=str)
parser.add_argument("script_args", nargs=REMAINDER)


def get_numa_cores():
    ret = []
    try:
        output = subprocess.check_output(['numactl', '--hardware']).decode("utf-8")
    except:
        return []
    lines = output.split('\n')
    for line in lines:
        if line.startswith('available:'):
            num_numas = int(line.split(' ')[1])
            break
    for numa in range(num_numas):
        for line in lines:
            if line.startswith(f'node {numa} cpus:'):
                cores = line.split(' ')[3:]
                ret.append([int(core) for core in cores])
    return ret

def parse_range(rng):
    try:
        value = int(rng)
        return range(value, value + 1)
    except ValueError:
        # value is not a single number
        parts = rng.split('-')
        if len(parts) != 2:
            raise ValueError("Bad range: '%s', range must be either a number or two number separated by dash" %
                             (rng, ))
        start = int(parts[0])
        end = int(parts[1])
        if start > end:
            raise ValueError("Bad range: '%s', range end must larger than or equal to start" % (rng, ))
        return range(start, end + 1)

def parse_range_list(range_str):
    number_list = []
    last = -1
    range_list = range_str.split(',')
    for sub_range in range_list:
        sub_number_list = parse_range(sub_range)
        if sub_number_list[0] <= last:
            raise ValueError(
                "Bad range: '%s', sub ranges must not overlap with each other and should be in ascend order" %
                (range_str, ))
        last = sub_number_list[-1]
        number_list.extend(sub_number_list)
    return number_list

def get_numactl_cmd(bind_core_list, num_local_procs, local_rank):
    numactl_cmd = []
    if bind_core_list is not None:
        core_list = parse_range_list(bind_core_list)
        total_cores = len(core_list)
    else:
        total_cores = psutil.cpu_count(logical=False)
        core_list = range(total_cores)
    cores_per_rank = total_cores // num_local_procs
    assert cores_per_rank >= 1, "At least one core needs to be assigned to each rank"
    core_list_for_rank = core_list[cores_per_rank * local_rank:cores_per_rank * (local_rank + 1)]
    numactl_cmd.append("numactl")

    # check if all cores belong to same numa, if true, bind process to that numa domain with -m parameter
    numa_cores = get_numa_cores()
    num_numas = len(numa_cores)

    numa_mode = "normal"

    non_empty_numa_list = []
    empty_numa_list = []
    previous_numa_cores = []
    numa_node_list = []
    numa_node_list_list = []
    for i in range(num_numas):
        # look for empty numa which is HBM numa
        if numa_cores[i] == []:
            empty_numa_list.append(i)
        else:
            non_empty_numa_list.append(i)

            # check for fakenuma
            if numa_cores[i] == previous_numa_cores:
                if numa_node_list == []:
                    #first duplication, add previous node into list
                    numa_node_list.append(i - 1)
                numa_node_list.append(i)
            else:
                if numa_node_list != []:
                    numa_node_list_list.append(numa_node_list)
                    numa_node_list = []
        previous_numa_cores = numa_cores[i]
    if numa_node_list != []:
        numa_node_list_list.append(numa_node_list)

    if empty_numa_list != [] and len(empty_numa_list) == len(non_empty_numa_list):
        numa_mode = "flat_hbm"
        numa_dict = dict(zip(non_empty_numa_list, empty_numa_list))
    elif numa_node_list_list != []:
        numa_mode = "fake"

    if numa_mode == "normal":
        for i in range(num_numas):
            if set(core_list_for_rank) <= set(numa_cores[i]):
                numactl_cmd.append("-m")
                numactl_cmd.append(f"{i}")
                break
    elif numa_mode == "flat_hbm":
        for i in range(num_numas):
            if set(core_list_for_rank) <= set(numa_cores[i]):
                numactl_cmd.append("-p")
                numactl_cmd.append(f"{numa_dict[i]}")
                break
    elif numa_mode == "fake":
        for i in range(num_numas):
            if set(core_list_for_rank) <= set(numa_cores[i]):
                for nodes in numa_node_list_list:
                    if i in nodes:
                        numactl_cmd.append("-m")
                        numactl_cmd.append(f"{','.join(map(str, nodes))}")
                        break
                # the following construct break the outer loop if inner loop breaks
                else:
                    continue
                break

    numactl_cmd.append("-C")
    last_core = core_list_for_rank[0]
    first_core = last_core
    core_list_str = f"{last_core}"
    for core_id in core_list_for_rank[1:]:
        if core_id == last_core + 1:
            last_core = core_id
            continue
        else:
            if first_core == last_core:
                core_list_str = f"{core_list_str},{core_id}"
            else:
                core_list_str = f"{core_list_str}-{last_core},{core_id}"
            first_core = core_id
            last_core = core_id
    if first_core != last_core:
        core_list_str = f"{core_list_str}-{last_core}"
    numactl_cmd.append(f"{core_list_str}")
    return cores_per_rank, numactl_cmd



def terminate_process_tree(pid):
    process = psutil.Process(pid)
    children = process.children(recursive=True)
    children.append(process)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(children, timeout=30)
    for p in alive:
        p.kill()


def main():

    args = parser.parse_args()
    world_size = args.world_size

    current_env = os.environ.copy()
    # set PyTorch distributed related environmental variables
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(world_size)
    current_env["LOCAL_SIZE"] = str(world_size)
    current_env["LOCAL_WORLD_SIZE"] = str(world_size)

    processes = []
    for rank in range(world_size):
        local_rank = rank
        current_env["RANK"] = str(rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        cmd = []
        if args.bind_cores_to_rank:
            cores_per_rank, numactl_cmd = get_numactl_cmd(args.bind_core_list, world_size, local_rank)
            current_env["OMP_NUM_THREADS"] = f"{cores_per_rank}"
            cmd = cmd + numactl_cmd
        cmd.append("python")
        cmd.append(args.script)
        cmd += args.script_args
        process = subprocess.Popen(cmd, env=current_env)
        print(f"process {rank} spawned with command: {cmd}", flush=True)
        processes.append(process)

    # handle processes
    sig_names = {2: "SIGINT", 15: "SIGTERM"}
    last_return_code = None

    def sigkill_handler(signum, frame):
        for process in processes:
            print(f"Killing subprocess {process.pid}", flush=True)
            try:
                terminate_process_tree(process.pid)
            except Exception:
                pass
        if last_return_code is not None:
            print(f"{cmd} exits with return code = {last_return_code}", flush=True)
            print(last_return_code, flush=True)
        if signum in sig_names:
            print(f"Main process received {sig_names[signum]}, exiting", flush=True)
        sys.exit(1)

    # pass SIGINT/SIGTERM to children if the parent is being terminated
    signal.signal(signal.SIGINT, sigkill_handler)
    signal.signal(signal.SIGTERM, sigkill_handler)

    alive_processes = set(processes)
    while len(alive_processes):
        finished_processes = []
        for process in alive_processes:
            if process.poll() is None:
                # the process is still running
                continue
            else:
                if process.returncode != 0:
                    last_return_code = process.returncode  # for sigkill_handler
                    sigkill_handler(signal.SIGTERM, None)  # not coming back
                else:
                    # exited cleanly
                    print(f"Process {process.pid} exits successfully.", flush=True)
                    finished_processes.append(process)
        alive_processes = set(alive_processes) - set(finished_processes)

        time.sleep(1)    
        

if __name__ =='__main__':
    main()
