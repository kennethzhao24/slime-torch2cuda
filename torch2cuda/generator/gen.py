import ast
from main_cp_sat_v1 import Graph
import json
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Type
import multiprocessing
from multiprocessing import Process, Lock, Value
import argparse
import random
import signal
import sys

from ops import *

# Set multiprocessing start method to 'spawn' for better CUDA compatibility
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

@dataclass(frozen=True)
class OpSpec:
    op_cls: Type[BasicOp]
    num_inputs: int

from main_cp_sat_v1 import DEFAULT_OP_SPECS

def remove_func_from_pytorch_code(pytorch_code: str, func_name: str):
    """
    Remove the <func_name> in <pytorch_code>.
    Return: pytorch_code_after_removal, removed_pytorch_func
    
    <pytorch_code_after_removal> is the pytorch code after removing the function <func_name>
    <removed_pytorch_func> is the code that being removed from pytorch_code

    Args:
    """
    if not isinstance(pytorch_code, str):
        raise TypeError("pytorch_code must be a string")
    if not isinstance(func_name, str):
        raise TypeError("func_name must be a string")

    try:
        tree = ast.parse(pytorch_code)
    except SyntaxError as exc:
        raise ValueError(f"pytorch_code is not valid python code: {exc}") from exc

    target_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            if node.lineno is None or node.end_lineno is None:
                raise ValueError(f"Cannot determine source range for function {func_name}")
            target_ranges.append((node.lineno, node.end_lineno))

    if not target_ranges:
        raise ValueError(f"Function {func_name} not found in provided code")

    lines = pytorch_code.splitlines()
    had_trailing_newline = pytorch_code.endswith("\n")
    removed_blocks: list[str] = []

    # Remove from bottom-up to keep indexes stable; trim blank lines just above the target.
    for start, end in sorted(target_ranges, key=lambda r: r[0], reverse=True):
        start_idx = start - 1
        while start_idx > 0 and lines[start_idx - 1].strip() == "":
            start_idx -= 1
        removed_slice = lines[start_idx:end]
        removed_blocks.append("\n".join(removed_slice) + ("\n" if had_trailing_newline else ""))
        del lines[start_idx:end]

    new_code = "\n".join(lines)
    if had_trailing_newline and new_code:
        new_code += "\n"
    removed_pytorch_func = "".join(reversed(removed_blocks))
    return new_code, removed_pytorch_func

def validate_sample(get_inputs, fused_operator):
    inputs = get_inputs()
    inputs = [
        x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs
    ]
    outputs = fused_operator(*inputs)
    
    # Check for NaN or Inf values in outputs
    has_nan_or_inf = False
    if isinstance(outputs, torch.Tensor):
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            has_nan_or_inf = True
    elif isinstance(outputs, (list, tuple)):
        for output in outputs:
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any() or torch.isinf(output).any():
                    has_nan_or_inf = True
                    break
    
    # Clean up
    del inputs, outputs
    torch.cuda.empty_cache()
    
    if has_nan_or_inf: raise ValueError("Output contains NaN or Inf")

def _generate_graph_and_source(
    min_flops,
    max_flops,
    num_nodes,
    node_op_types: Optional[List[Type[BasicOp]]] = None,
    level: int = 0,
):
    while True:
        try:
            g = Graph(min_flops = min_flops, max_flops = max_flops)
            g.generate_random_graph(num_nodes, node_op_types)
            src = g.to_torch()
            context = {}
            # output syntax check
            exec(src, context)

        except Exception as e:
            if e.args and e.args[0] == "Not feasible":
                if min_flops == 0:
                    raise ValueError(f"No sample found {node_op_types}, min_flops = 0")
                min_flops = min_flops // 2
            else:
                raise e
        else:
            return g, src

def _shape_tuple(shape: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(dim) for dim in shape)

def _render_get_outputs(outputs, tensor_names: Dict[int, str]) -> str:
    lines = ["def get_outputs():"]
    if not outputs:
        lines.append("    return []")
        return "\n".join(lines)

    for edge in outputs:
        shape_tuple = _shape_tuple(getattr(edge, "shape", []))
        dtype_name = format_dtype(getattr(edge, "dtype", None))
        dtype_arg = f", dtype={dtype_name}" if dtype_name else ""
        lines.append(f"    {tensor_names[edge.id]} = torch.empty({shape_tuple}{dtype_arg})")

    output_names = [tensor_names[edge.id] for edge in outputs]
    lines.append(f"    return [{', '.join(output_names)}]")
    return "\n".join(lines)

def get_sample(min_flops, max_flops, num_nodes, node_op_types: Optional[List[Type[BasicOp]]] = None, level: int = 0):
    g, src = _generate_graph_and_source(min_flops, max_flops, num_nodes, node_op_types, level)
    return {
        "pytorch": src,
        "difficulty": level 
    }

def get_sample_with_io_metadata(
    min_flops,
    max_flops,
    num_nodes,
    node_op_types: Optional[List[Type[BasicOp]]] = None,
    level: int = 0,
):
    g, src = _generate_graph_and_source(min_flops, max_flops, num_nodes, node_op_types, level)
    pytorch_code, get_inputs_str = remove_func_from_pytorch_code(src, "get_inputs")
    input_nodes, outputs, tensor_names = g.io_tensors()
    input_shapes = [_shape_tuple(node.out.shape) for node in input_nodes]
    output_shapes = [_shape_tuple(edge.shape) for edge in outputs]
    get_outputs_str = _render_get_outputs(outputs, tensor_names)
    return {
        "pytorch": pytorch_code,
        "difficulty": level,
        "input_tensor_shapes": input_shapes,
        "output_tensor_shapes": output_shapes,
        "get_inputs": get_inputs_str,
        "get_outputs": get_outputs_str,
    }

def get_operator_combination_for_level(level: int, idx: int) -> List[Type[BasicOp]]:
    """Get the operator combination for a given level and index."""
    num_ops = len(DEFAULT_OP_SPECS)
    
    if level == 1:
        batch_idx = idx % num_ops
        return [DEFAULT_OP_SPECS[batch_idx].op_cls]
    elif level == 2:
        batch_idx = idx % (num_ops * num_ops)
        op1_idx = batch_idx // num_ops
        op2_idx = batch_idx % num_ops
        return [DEFAULT_OP_SPECS[op1_idx].op_cls, DEFAULT_OP_SPECS[op2_idx].op_cls]
    else:
        return [random.choice(DEFAULT_OP_SPECS).op_cls for _ in range(level)]

def worker_process(worker_id, start_idx, end_idx, output_file, file_lock, progress_counter, total_count, level, stop_flag):
    local_count = 0
    max_retries = 10  # Maximum number of retries with random combinations
    
    for idx in range(start_idx, end_idx):
        # Check if we should stop
        with stop_flag.get_lock():
            if stop_flag.value:
                print(f"Worker {worker_id} received stop signal, exiting...")
                return
        retry_count = 0
        success = False
        
        while not success and retry_count <= max_retries:
            # Check if we should stop before each retry
            with stop_flag.get_lock():
                if stop_flag.value:
                    print(f"Worker {worker_id} received stop signal, exiting...")
                    return
            
            try:
                # Get operator combination for this level and index
                # On first try, use deterministic combination; on retries, use random
                if retry_count == 0:
                    op_types = get_operator_combination_for_level(level, idx)
                else:
                    # Retry with random combination
                    op_types = [random.choice(DEFAULT_OP_SPECS).op_cls for _ in range(level)]
                
                num_nodes = len(op_types)
                src = get_sample(2**34, 2**35, num_nodes, op_types, level)
                # Write to file with lock
                with file_lock:
                    with open(output_file, "a") as f:
                        json.dump(src, f)
                        f.write('\n')
                        f.flush()
                    # Update progress counter
                    with progress_counter.get_lock():
                        progress_counter.value += 1
                        current_progress = progress_counter.value
                local_count += 1
                if local_count % 10 == 0:  # Print progress every 10 samples
                    print(f"Worker {worker_id}: Generated {local_count}/{end_idx - start_idx} samples (Total: {current_progress}/{total_count})")
                success = True
            except Exception as e:
                retry_count += 1
                if retry_count == 1:
                    print(f"Worker {worker_id} error at index {idx}: {e}")
                if retry_count <= max_retries:
                    print(f"Worker {worker_id} retrying index {idx} with random combination (attempt {retry_count}/{max_retries})")
                else:
                    print(f"Worker {worker_id} failed at index {idx} after {max_retries} retries: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate tensor computation graphs')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file path (e.g., random_torch_v2_l1.jsonl)')
    parser.add_argument('--level', type=int, required=True,
                        help='Generation level')
    parser.add_argument('--target_count', type=int, default=10000,
                        help='Target number of samples to generate (default: 10000)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index to generate (default: 0)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers to use (default: 1)')
    
    args = parser.parse_args()
    
    output_file = args.output_file
    level = args.level
    target_count = args.target_count
    
    num_ops = len(DEFAULT_OP_SPECS)
    
    # Calculate target_count based on level
    if level == 1:
        # Level 1: Single operator, batch size 47 (one for each operator)
        # Keep uncompleted batch if exists
        batch_size = num_ops
        num_complete_batches = target_count // batch_size
        remainder = target_count % batch_size
        if num_complete_batches == 0 and remainder == 0:
            raise ValueError(f"Level 0 requires at least 1 sample. Got {args.target_count}.")
        if num_complete_batches > 0:
            print(f"Level 0: Generating {num_complete_batches} complete batch(es) ({num_complete_batches * batch_size} samples)")
        if remainder > 0:
            print(f"Level 0: Plus 1 incomplete batch with {remainder} samples")
        print(f"Level 0: Total {target_count} single operator samples from {num_ops} operators")
    elif level == 2:
        batch_size = num_ops * num_ops
        num_complete_batches = target_count // batch_size
        remainder = target_count % batch_size
        if num_complete_batches == 0 and remainder == 0:
            raise ValueError(f"Level 1 requires at least 1 sample. Got {args.target_count}.")
        if num_complete_batches > 0:
            print(f"Level 1: Generating {num_complete_batches} complete batch(es) ({num_complete_batches * batch_size} samples)")
        if remainder > 0:
            print(f"Level 1: Plus 1 incomplete batch with {remainder} samples")
        print(f"Level 1: Total {target_count} combinations of 2 operators from {num_ops} operators")
    else:
        print(f"Level {level}: Generating {target_count} samples with {level} operators each (sampled)")

    # Get number of available GPUs
    num_workers = args.num_workers

    # Calculate samples per GPU
    samples_per_worker = target_count // num_workers
    remainder = target_count % num_workers

    print(f"Splitting {target_count} samples across {num_workers} workers: {samples_per_worker} samples per worker" + 
          (f" (with {remainder} extra samples distributed to first {remainder} workers)" if remainder > 0 else ""))

    # Create output file (empty it first)
    with open(output_file, "w") as f:
        pass

    # Create shared lock and progress counter
    file_lock = Lock()
    progress_counter = Value('i', 0)
    stop_flag = Value('i', 0)  # Flag to signal workers to stop

    # Create and start worker processes
    processes = []
    start_idx = args.start_idx

    for worker_id in range(num_workers):
        end_idx = start_idx + samples_per_worker + (1 if worker_id < remainder else 0)
        
        p = Process(target=worker_process, args=(worker_id, start_idx, end_idx, output_file, file_lock, progress_counter, target_count, level, stop_flag))
        p.start()
        processes.append(p)
        print(f"Started process for worker {worker_id}: generating samples {start_idx} to {end_idx-1} ({end_idx - start_idx} samples)")
        start_idx = end_idx

    def signal_handler(sig, frame):
        """Handle Ctrl-C by stopping all worker processes."""
        print("\n\nReceived interrupt signal (Ctrl-C). Stopping all workers...")
        with stop_flag.get_lock():
            stop_flag.value = 1
        
        # Terminate all processes
        for p in processes:
            if p.is_alive():
                p.terminate()
        
        # Wait for processes to terminate (with timeout)
        for p in processes:
            p.join(timeout=2)
            if p.is_alive():
                print(f"Force killing process {p.pid}...")
                p.kill()
                p.join()
        
        print(f"\nStopped. Generated {progress_counter.value} samples in {output_file}")
        sys.exit(0)

    # Register signal handler after processes are created
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Wait for all processes to complete
    print(f"\nWaiting for all {num_workers} workers to complete generation...")
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        # This should be caught by signal handler, but just in case
        signal_handler(signal.SIGINT, None)

    print(f"\nGeneration complete! Generated {progress_counter.value} samples in {output_file}")