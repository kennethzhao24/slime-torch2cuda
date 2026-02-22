'''
Usage:
python run_ray_rollout_new.py \
    --concurrency <concurrency_int> \
    --tensor_parallel_size <tensor_parallel_size_int> \
    --model_path <model_path_str> \
    --num_rollouts <num_rollouts_int> \
    --input_fn <input_fn_str> \
    --output_dir <output_dir_str>
    
Run rollout for model <model_path_str> on given input file <input_fn_str>, write to <output_dir_str>.

concurrency: how many concurrent queries. Default value 32.
tensor_parallel_size: the tp of the model
model_path: A huggingface mode id or a local path to huggingface checkpoint.

input_fn_str: A jsonl file. Each line is a json string of a python dict. Format:
```
{
  'pytorch_code': <pytorch_code_str>,
  'pytorch_inputs': <pytorch_inputs_str>,
  'pytorch': <pytorch_str>,
}
```

'''

import sys
import atexit
from pathlib import Path
from collections import deque, defaultdict

from utils import load_jsonl, remove_func_from_pytorch_code


sys.path.append(str(Path(__file__).resolve().parent))

import argparse
import json
import os
import time
import subprocess
import requests
import signal

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
ray.init(address="auto")

os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', '/dev/null')
temperature = 1.0

def get_ray_resource_list():
    '''
    Return a list of nodes and its resources.
    Return:
    [
        {
            'ip': <node1_ip>,
            'num_cpus': <node1_cpus>,
            'num_gpus': <node1_gpus>,
        }
    ]
    '''
    nodes = []
    for node in ray.nodes():
        if not node.get("Alive"):
            continue
        resources = node.get("Resources")
        if resources is None:
            raise KeyError("Resources are missing in Ray node metadata.")
        if "CPU" not in resources:
            raise KeyError("CPU resource not found in Ray node metadata.")
        if "GPU" not in resources:
            raise KeyError("GPU resource not found in Ray node metadata.")
        ip_addr = node.get("NodeManagerAddress")
        if ip_addr is None:
            raise KeyError("NodeManagerAddress missing in Ray node metadata.")
        nodes.append(
            {
                "ip": ip_addr,
                "num_cpus": int(resources["CPU"]),
                "num_gpus": int(resources["GPU"]),
            }
        )
    return nodes

@ray.remote
def start_a_single_sglang_server(model_id, tp, nnodes, MASTER_IP, node_id, port, dist_port, log_dir):
    if not isinstance(dist_port, int):
        raise TypeError("dist_port must be an int")
    if dist_port <= 0:
        raise ValueError("dist_port must be positive")
    if not isinstance(log_dir, (str, Path)):
        raise TypeError("log_dir must be a str or Path")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    node_ip = ray.util.get_node_ip_address()
    log_path = log_dir / f"sglang_{node_ip}_{port}.log"
    script_dir = Path(__file__).resolve().parent
    cmd = [
        "bash",
        str(script_dir / "start_sglang_server.sh"),
        str(model_id),
        str(tp),
        str(nnodes),
        str(MASTER_IP),
        str(node_id),
        str(port),
        str(dist_port),
    ]
    print(f'run bash cmd: {cmd} on node {node_ip}, log={log_path}')
    with open(log_path, "a") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,  # allow killing the full group
        )
    return {"pid": proc.pid, "log_path": str(log_path)}


@ray.remote
def stop_sglang_server(pid):
    if not isinstance(pid, int):
        raise TypeError("pid must be an int")
    if pid <= 0:
        raise ValueError("pid must be positive")
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    return True


@ray.remote
def is_process_alive(pid):
    if not isinstance(pid, int):
        raise TypeError("pid must be an int")
    if pid <= 0:
        raise ValueError("pid must be positive")
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def launch_sglang_on_ray_cluster(node_resource_list, num_cpus_per_node, num_gpus_per_node, model_id, tp, log_dir):
    '''
    Launch sglang servers on ray cluster.
    
    ## Single node mode
    
    When tensor_parallel_size <= num_gpus_per_node, start {num_gpus_per_node //tensor_parallel_size} sglang servers per node.
    Assert num_gpus_per_node // tensor_parallel_size * tensor_parallel_size == num_gpus_per_node.
    For the i-th sglang server, start it using this call:
    start_a_single_sglang_server.remote(model_id, tp, nnodes=1, MASTER_IP=node["ip"], node_id=0, port=30000+i)
    After call, return immediately, let the server run in background.
    
    Return:
    
    A list of servers and port:
    [
        {
            'ip': <sglang_server_0_ip>,
            'port': <sglang_server_0_port>,
        },
        {
            'ip': <sglang_server_1_ip>,
            'port': <sglang_server_1_port>,
        },
        ....
    ]
    Note that a node may have multiple sglang server, each using a different port but same ip.
    
    ## Multi-node mode
    
    When tensor_parallel_size > num_gpus_per_node, start num_sglang_servers={len(node_resource_list) * num_gpus_per_node // tensor_parallel_size} sglang servers.
    Assert tensor_parallel_size // num_gpus_per_node * num_gpus_per_node == tensor_parallel_size. 
    
    In this mode, each sglang server use n=tensor_parallel_size // num_gpus_per_node nodes. For a group of n nodes, denote the 0-th node as head node, and remaining node as worker node.
    Denote the head node ip as <head_node_ip>. Within a group, start the sglang server on each node by:
    start_a_single_sglang_server(model_id, tp, nnodes=n, MASTER_IP=<head_node_ip>, node_id=<node_id_in_this_group>, port=30000+{node_id_in_this_group})
    
    node_id_in_this_group starts from 0. Head node in a group is always node_ip=0.
    
    Return:
    
    A list of head nodes ip and ports:
    [
        {
            'ip': <head_node_0_ip>,
            'port': <head_node_0_port>,
        },
        {
            'ip': <head_node_1_ip>,
            'port': <head_node_1_port>,
        },
        ...
    ]
    
    '''
    
    if tp <= 0:
        raise ValueError("tp must be positive.")
    if num_gpus_per_node <= 0:
        raise ValueError("num_gpus_per_node must be positive.")
    if not isinstance(log_dir, (str, Path)):
        raise TypeError("log_dir must be a str or Path")
    ip_to_node_id = {
        node["NodeManagerAddress"]: node["NodeID"]
        for node in ray.nodes()
        if node.get("Alive")
    }
    server_list = []
    start_jobs = []
    if tp <= num_gpus_per_node:
        if num_gpus_per_node % tp != 0:
            raise ValueError(
                f"num_gpus_per_node={num_gpus_per_node} is not divisible by tp={tp}."
            )
        servers_per_node = num_gpus_per_node // tp
        for node in node_resource_list:
            node_id = ip_to_node_id.get(node["ip"])
            if node_id is None:
                raise KeyError(f"Cannot find Ray node id for ip {node['ip']}.")
            for local_idx in range(servers_per_node):
                port = 30000 + local_idx
                dist_port = 5000 + local_idx  # avoid port clash when multiple servers per node
                pid_ref = start_a_single_sglang_server.options(
                    num_cpus=max(1, num_cpus_per_node // servers_per_node),
                    num_gpus=tp,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    ),
                ).remote(model_id, tp, 1, node["ip"], 0, port, dist_port, log_dir)
                start_jobs.append(
                    {"node_id": node_id, "ip": node["ip"], "port": port, "pid_ref": pid_ref}
                )
                server_list.append({"ip": node["ip"], "port": port})
    else:
        if tp % num_gpus_per_node != 0:
            raise ValueError(
                f"tp={tp} is not divisible by num_gpus_per_node={num_gpus_per_node}."
            )
        nodes_per_server = tp // num_gpus_per_node
        num_servers = len(node_resource_list) * num_gpus_per_node // tp
        for server_idx in range(num_servers):
            group_start = server_idx * nodes_per_server
            group_nodes = node_resource_list[group_start : group_start + nodes_per_server]
            if len(group_nodes) < nodes_per_server:
                raise ValueError("Not enough nodes to form a complete server group.")
            head_ip = group_nodes[0]["ip"]
            dist_port = 5000 + server_idx  # ensure unique init port per server group
            for node_rank, node in enumerate(group_nodes):
                node_id = ip_to_node_id.get(node["ip"])
                if node_id is None:
                    raise KeyError(f"Cannot find Ray node id for ip {node['ip']}.")
                port = 30000
                pid_ref = start_a_single_sglang_server.options(
                    num_cpus=num_cpus_per_node,
                    num_gpus=num_gpus_per_node,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=False
                    ),
                ).remote(model_id, tp, nodes_per_server, head_ip, node_rank, port, dist_port, log_dir)
                start_jobs.append(
                    {"node_id": node_id, "ip": node["ip"], "port": port, "pid_ref": pid_ref}
                )
            server_list.append({"ip": head_ip, "port": 30000})
    proc_info_list = []
    proc_results = ray.get([job["pid_ref"] for job in start_jobs])
    for job, res in zip(start_jobs, proc_results):
        if not isinstance(res, dict) or "pid" not in res or "log_path" not in res:
            raise TypeError("start_a_single_sglang_server must return dict with pid and log_path")
        proc_info_list.append(
            {
                "node_id": job["node_id"],
                "ip": job["ip"],
                "port": job["port"],
                "pid": res["pid"],
                "log_path": res["log_path"],
            }
        )
    return server_list, proc_info_list
    

def test_sglang_server_alive(sglang_server_ip_port_list, model_id, interval_sec=60, timeout=180, proc_info_list=None, max_wait_sec=1800):
    if not isinstance(sglang_server_ip_port_list, list):
        raise TypeError("sglang_server_ip_port_list must be a list")
    if not sglang_server_ip_port_list:
        raise ValueError("sglang_server_ip_port_list cannot be empty")
    for item in sglang_server_ip_port_list:
        if not isinstance(item, dict) or "ip" not in item or "port" not in item:
            raise TypeError("Each server entry must be a dict with ip and port")
    if not isinstance(model_id, str):
        raise TypeError("model_id must be a string")
    if proc_info_list is not None and not isinstance(proc_info_list, list):
        raise TypeError("proc_info_list must be a list when provided")
    if max_wait_sec <= 0:
        raise ValueError("max_wait_sec must be positive")

    ready_state = {
        f"{srv['ip']}:{srv['port']}": {
            "ready": False,
        }
        for srv in sglang_server_ip_port_list
    }
    proc_state = {}
    if proc_info_list is not None:
        for proc in proc_info_list:
            if not isinstance(proc, dict):
                raise TypeError("Each process entry must be a dict")
            for key in ("pid", "node_id", "ip", "port", "log_path"):
                if key not in proc:
                    raise KeyError(f"Process entry missing {key}")
            proc_state.setdefault(f"{proc['ip']}:{proc['port']}", []).append(proc)
    pending_servers = list(sglang_server_ip_port_list)
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {"role": "user", "content": "Just a simple ping. You should simply say 'I am ready'. Do not think. "}
        ],
    }
    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    attempt = 0
    while True:
        if not pending_servers:
            break

        attempt += 1
        print(f"[sglang ready check] attempt={attempt} pending={len(pending_servers)}")
        to_remove = []
        for srv in pending_servers:
            key = f"{srv['ip']}:{srv['port']}"
            state = ready_state[key]
            try:
                url = f"http://{srv['ip']}:{srv['port']}/v1/chat/completions"
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                is_ready = isinstance(data, dict) and "choices" in data
            except Exception as exc:
                is_ready = False
                print(f"[sglang ready check] {key} not ready yet: {exc}")

            if is_ready and not state["ready"]:
                state["ready"] = True
                to_remove.append(srv)

        if to_remove:
            for srv in to_remove:
                pending_servers.remove(srv)
        status_lines = []
        for srv in sglang_server_ip_port_list:
            key = f"{srv['ip']}:{srv['port']}"
            state = ready_state[key]
            status = "Ready" if state["ready"] else "Not ready"
            status_lines.append(f"node {key}: {status}")
        print("\n".join(status_lines))
        elapsed = time.time() - start_time
        if pending_servers and elapsed >= max_wait_sec:
            failure_lines = []
            dead_proc_lines = []
            if proc_state:
                check_tasks = []
                task_meta = []
                for proc_list in proc_state.values():
                    for proc in proc_list:
                        check_tasks.append(
                            is_process_alive.options(
                                scheduling_strategy=NodeAffinitySchedulingStrategy(
                                    node_id=proc["node_id"], soft=False
                                )
                            ).remote(proc["pid"])
                        )
                        task_meta.append(proc)
                results = ray.get(check_tasks)
                for meta, alive in zip(task_meta, results):
                    status = "alive" if alive else "not running"
                    dead_proc_lines.append(
                        f"node {meta['ip']} port {meta['port']} pid {meta['pid']} ({status}) log={meta['log_path']}"
                    )
            pending_keys = [f"{srv['ip']}:{srv['port']}" for srv in pending_servers]
            failure_lines.append(
                f"SGLang servers failed to become ready within {int(max_wait_sec)}s. Pending: {', '.join(pending_keys)}"
            )
            if dead_proc_lines:
                failure_lines.append("Process state:")
                failure_lines.extend(dead_proc_lines)
            raise RuntimeError("\n".join(failure_lines))
        time.sleep(interval_sec)


def stop_all_sglang_servers(proc_info_list):
    if not isinstance(proc_info_list, list):
        raise TypeError("proc_info_list must be a list")
    if not proc_info_list:
        return
    kill_tasks = []
    for proc in proc_info_list:
        if not isinstance(proc, dict):
            raise TypeError("Each process entry must be a dict")
        if "pid" not in proc or "node_id" not in proc:
            raise KeyError("Process entry missing pid or node_id")
        pid = proc["pid"]
        node_id = proc["node_id"]
        kill_tasks.append(
            stop_sglang_server.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                )
            ).remote(pid)
        )
    ray.get(kill_tasks)

def get_llm_response(ip, port, prompt):
    '''
    Submit a query prompt to llm engine server at ip:port, return the sglang server response (exclude the prompt).
    refer to llm_engine.py for how to sumit a query prompt to an openai compatible server.
    '''
    if not isinstance(ip, str):
        raise TypeError("ip must be a string")
    if not isinstance(port, (str, int)):
        raise TypeError("port must be a string or int")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not prompt:
        raise ValueError("prompt must be non-empty")
    if not isinstance(temperature, (int, float)):
        raise TypeError("temperature must be a number")

    url = f"http://{ip}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_path,
        "messages": [
            {"role": "system", "content": 'you are a helpful AI assitant.'},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=1800)
        resp.raise_for_status()
        data = resp.json()
        if "choices" not in data:
            raise KeyError("Missing 'choices' in response")
        choices = data["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("'choices' must be a non-empty list")
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            raise TypeError("Each choice must be a dict")
        if "message" in choice0:
            message = choice0["message"]
            if not isinstance(message, dict) or "content" not in message:
                raise KeyError("Missing 'message.content' in response choice")
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError("message.content must be a string")
            return content
        if "text" in choice0:
            text = choice0["text"]
            if not isinstance(text, str):
                raise TypeError("choice.text must be a string")
            return text
        raise KeyError("Unsupported response schema: missing message.content or text")
    except Exception as exc:
        print(f"Failed to get llm response from {url}: {exc}")
        return None

with open("prompt_template/convert_pytorch_to_cuda_prompt.txt") as fid:
    global_prompt_template=fid.read()

def prepare_prompt(sample):
    '''
    sample is {
        "pytorch_code": <pytorch_code_str>,
        "pytorch_inputs": <pytorch_inputs_str>,
    }
    
    '''
    pytorch_code = sample['pytorch_code']
    pytorch_inputs = sample['pytorch_inputs']
    prompt = global_prompt_template.format(pytorch_code=pytorch_code, pytorch_inputs=pytorch_inputs)
    return prompt
    
def do_main_rollout():
    the_pytorch_script_to_rollout_list = load_jsonl(input_fn)
    new_list = []
    for job_id, input_json in enumerate(the_pytorch_script_to_rollout_list):
        if "pytorch_code" in input_json and "pytorch_inputs" in input_json:
            pytorch_code, pytorch_inputs = input_json["pytorch_code"], input_json["pytorch_inputs"]
        else:
            pytorch_code_and_inputs = input_json["pytorch"]
            pytorch_code, pytorch_inputs = remove_func_from_pytorch_code(
                pytorch_code_and_inputs,
                func_name="get_inputs",
            )
            pytorch_inputs = "import torch\n" + pytorch_inputs
        extra_fields = {
            k: v
            for k, v in input_json.items()
            if k not in ("pytorch_code", "pytorch_inputs", "pytorch")
        }
        new_list.append({
            "pytorch_code": pytorch_code,
            "pytorch_inputs": pytorch_inputs,
            **extra_fields,
        })
    the_pytorch_script_to_rollout_list = new_list
    
    print('Working on ' + str(output_dir))
    
    final_output_fn = os.path.join(output_dir, "final_rollout_output.jsonl")
    final_output_done_fn = os.path.join(output_dir, "final_rollout_output.done")
    if os.path.isfile(final_output_done_fn):
        print(f"skip {final_output_fn}")
        return
    
    if os.path.isfile(final_output_fn):
        rollouted_samples = load_jsonl(final_output_fn)
    else:
        rollouted_samples = []
        
    rollouted_counts = defaultdict(int)
    for item in rollouted_samples:
        key = item["pytorch_code"] + item["pytorch_inputs"]
        rollouted_counts[key] += 1
    print(f"found {sum(rollouted_counts.values())} existing rollouts across {len(rollouted_counts)} samples")
    
    # filter out fully processed samples and track remaining rollouts per sample
    pending_samples = []
    for pytorch_script_to_rollout in the_pytorch_script_to_rollout_list:
        pytorch_code = pytorch_script_to_rollout["pytorch_code"]
        pytorch_inputs = pytorch_script_to_rollout["pytorch_inputs"]
        key = pytorch_code + pytorch_inputs
        existing = rollouted_counts.get(key, 0)
        remaining = max(0, num_rollouts - existing)
        if remaining <= 0:
            continue
        pending_samples.append((pytorch_script_to_rollout, remaining))
    the_pytorch_script_to_rollout_list = [s for s, _ in pending_samples]
    
    @ray.remote(num_cpus=0)
    def _rollout_once(ip, port, sample, rollout_id):
        prompt = prepare_prompt(sample)
        response = get_llm_response(ip, port, prompt)
        if response is None:
            return None
        result = dict(sample)
        result["response"] = response
        result["rollout_id"] = rollout_id
        return result
    
    rollout_jobs = deque()
    for sample, remaining in pending_samples:
        key = sample["pytorch_code"] + sample["pytorch_inputs"]
        start_id = rollouted_counts.get(key, 0)
        for rollout_id in range(start_id, start_id + remaining):
            rollout_jobs.append((sample, rollout_id))
    
    total_rollout_jobs = len(rollout_jobs)
    total_samples_written = 0
    server_inflight = [0 for _ in sglang_server_ip_port_list]
    inflight = {}
    start_time = time.time()
    last_eta_log = 0.0
    
    print(f"Submitting {len(rollout_jobs)} rollout jobs across {len(sglang_server_ip_port_list)} servers.")
    with open(final_output_fn, "a") as fout:
        while rollout_jobs or inflight:
            now = time.time()
            if now - last_eta_log >= 5.0 and total_rollout_jobs > 0:
                processed = total_samples_written
                remaining = total_rollout_jobs - processed
                rate = processed / max(1e-6, now - start_time)
                eta_sec = remaining / max(1e-6, rate) if processed > 0 else float("inf")
                eta_str = f"{int(eta_sec)}s" if eta_sec != float("inf") else "unknown"
                print(f"progress: {processed}/{total_rollout_jobs}, remaining={remaining}, eta={eta_str}")
                last_eta_log = now
            for server_idx, server in enumerate(sglang_server_ip_port_list):
                while rollout_jobs and server_inflight[server_idx] < concurrency:
                    sample, rollout_id = rollout_jobs.popleft()
                    ref = _rollout_once.remote(server["ip"], server["port"], sample, rollout_id)
                    inflight[ref] = server_idx
                    server_inflight[server_idx] += 1
            if not inflight:
                break
            ready_refs, _ = ray.wait(list(inflight.keys()), num_returns=1)
            for ref in ready_refs:
                server_idx = inflight.pop(ref)
                server_inflight[server_idx] -= 1
                try:
                    result = ray.get(ref)
                except Exception as exc:
                    print(f"rollout task failed: {exc}")
                    continue
                if result is None:
                    continue
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                total_samples_written += 1
    total_lines = len(rollouted_samples) + total_samples_written
    with open(final_output_done_fn, "w") as fid:
        fid.write(json.dumps({
            "total_run_time": time.time() - start_time,
            "total_samples_written": total_samples_written,
            "total_lines_written": total_lines,
        }))
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run rollout for a model on the provided input file and store outputs."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Number of concurrent queries to run (default: 32).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        required=True,
        help="Tensor parallel size for the model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument(
        "--num_rollouts",
        type=int,
        required=True,
        help="Number of rollouts to generate per input.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0).",
    )
    parser.add_argument(
        "--input_fn",
        type=str,
        required=True,
        help="Path to the input jsonl file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write rollout outputs.",
    )
    args = parser.parse_args()
    
    concurrency = args.concurrency
    tensor_parallel_size = args.tensor_parallel_size
    model_path = args.model_path
    num_rollouts = args.num_rollouts
    temperature = args.temperature
    input_fn = args.input_fn
    output_dir = args.output_dir
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_rollout_output_done_fn = os.path.join(str(output_dir), 'final_rollout_output.done')
    if os.path.isfile(final_rollout_output_done_fn):
        print(f'skip {final_rollout_output_done_fn}')
        exit()

    node_resource_list = get_ray_resource_list()
    if not node_resource_list:
        raise ValueError("No alive Ray nodes found in the cluster.")
    
    # ensure all nodes have same number of GPUs an CPUs.
    num_cpus_per_node = node_resource_list[0]['num_cpus']
    num_gpus_per_node = node_resource_list[0]['num_gpus']
    for node in node_resource_list[1:]:
        assert node['num_cpus'] == num_cpus_per_node, f'node {node["ip"]} has {node["num_cpus"]} CPUs but num_cpus_per_node={num_cpus_per_node}.'
        assert node['num_gpus'] == num_gpus_per_node, f'node {node["ip"]} has {node["num_gpus"]} CPUs but num_cpus_per_node={num_gpus_per_node}.'
        
    print(f'Starting SGLanger server on {len(node_resource_list)} nodes.')
    proc_info_list = []
    sglang_server_ip_port_list, proc_info_list = launch_sglang_on_ray_cluster(node_resource_list=node_resource_list,
                                 num_cpus_per_node=num_cpus_per_node,
                                 num_gpus_per_node=num_gpus_per_node,
                                 model_id=args.model_path,
                                 tp=args.tensor_parallel_size,
                                 log_dir=output_dir / "logs")

    cleanup_state = {"done": False}

    def cleanup_servers():
        if cleanup_state["done"]:
            return
        cleanup_state["done"] = True
        try:
            stop_all_sglang_servers(proc_info_list)
        except Exception as exc:
            print(f"Failed to stop all sglang servers cleanly: {exc}")

    def handle_signal(signum, frame):
        print(f"Received signal {signum}, stopping sglang servers...")
        cleanup_servers()
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    atexit.register(cleanup_servers)

    print(f'sglang_server_ip_port_list={sglang_server_ip_port_list}')
    print("sglang server logs:")
    for proc in proc_info_list:
        print(f"  node={proc['ip']} pid={proc['pid']} log={proc['log_path']}")

    test_sglang_server_alive(
        sglang_server_ip_port_list,
        args.model_path,
        proc_info_list=proc_info_list,
    )

    print(f'All server is up. Waiting your inputs')
    try:
        do_main_rollout()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down servers...")
    finally:
        cleanup_servers()