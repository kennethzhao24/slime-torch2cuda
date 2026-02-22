from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Union, Tuple
import json
import re

def extract_torch_ops_from_code(code: str) -> List[str]:
    """
    Extract torch operations from Python code string.
    Matches patterns like torch.operation_name(...) or torch.nn.functional.avg_pool3d(...)
    Captures the full path after torch. including dots (e.g., 'nn.functional.avg_pool3d', 'add', 'matmul')
    Excludes tensor creation ops and dtype constants.
    """
    # Operations to exclude (tensor creation, random ops, dtype constants, etc.)
    excluded_ops = {
        # Tensor creation
        'randn', 'rand', 'randint', 'randn_like', 'rand_like', 'randint_like',
        'zeros', 'zeros_like', 'ones', 'ones_like', 'empty', 'empty_like',
        'full', 'full_like', 'arange', 'linspace', 'logspace', 'eye',
        'tensor', 'as_tensor', 'from_numpy', 'frombuffer',
        
        # Dtype constants
        'float32', 'float64', 'float16', 'bfloat16',
        'int32', 'int64', 'int16', 'int8',
        'uint8', 'bool', 'complex64', 'complex128',
        'float', 'double', 'half', 'long', 'int', 'short', 'byte',
        
        # Device
        'device',
        
        # Random number generator
        'manual_seed', 'initial_seed', 'seed', 'get_rng_state', 'set_rng_state',
    }
    
    # Pattern to match torch. followed by one or more identifiers separated by dots
    pattern = r'torch\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
    matches = re.findall(pattern, code)
    
    # Filter out excluded operations
    filtered_matches = ['torch.' + op for op in matches if op not in excluded_ops]
    
    return filtered_matches

def process_jsonl_file(file_path: Union[str, Path]) -> Tuple[Counter, List[Tuple[int, ...]], List[int]]:
    """
    Process a JSONL file and extract torch operations, tensor dimensions, and levels.
    Expected format: each line is a JSON object with 'pytorch' field
    Returns: (op_counter, tensor_dimensions, levels)
    """
    op_counter = Counter()
    levels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                # Extract level/difficulty if present
                if 'difficulty' in item:
                    levels.append(item['difficulty'])
                
                # Check for pytorch field
                if 'pytorch' in item and item['pytorch']:
                    code = item['pytorch']
                    ops = extract_torch_ops_from_code(code)
                    op_counter.update(ops)
            except json.JSONDecodeError:
                continue
    
    return op_counter, levels