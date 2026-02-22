# Tensor Computation Graph Generator

This generator creates synthetic PyTorch tensor computation graphs for training and evaluating Triton kernel generation models. It uses constraint programming (CP-SAT) to generate valid tensor operations with proper shape constraints and FLOPS requirements.

## Overview

The generator produces PyTorch code that implements fused tensor operations, which can be used as training data for learning to generate optimized Triton kernels. Each generated sample includes:

- A `fused_operator` function that performs the computation
- A `get_inputs` function that generates input tensors
- A difficulty level (0-4) indicating the complexity

## Update Log
### V3
- 
### V4
- Add new operators: **ConvTransposeNd**, **Norm**, **Triu**, **Tril**, **Clamp**, **Abs**.
- Fix the randomness issue of **ConvNd** and **ConvTransposeNd**.
- Fix the issue that standard output contains inf or nan numbers.
- Add new auto parameters to **ConvNd** and **ConvTransposeNd**: `stride`, `padding`, `dilation`, `group`
- Fix the generator sometimes can't generate `dim > 1` or `group > 1`.

## Features

- **47+ Tensor Operations**: Supports a wide variety of operations including elementwise ops, reductions, activations, normalizations, convolutions, and more
- **Constraint-Based Generation**: Uses Google OR-Tools CP-SAT solver to ensure valid tensor shapes and operations
- **Multi-Level Difficulty**: 5 difficulty levels (0-4) with increasing complexity
- **Multi-GPU Support**: Parallel generation across multiple GPUs
- **FLOPS Control**: Configurable minimum and maximum FLOPS requirements

## Installation

The generator requires the following dependencies:

```bash
pip install torch ortools
```

## Usage

### Basic Usage

```bash
python generator/gen.py --output_file output.jsonl --level 1 --target_count 10000
```

### Arguments

- `--output_file` (required): Path to the output JSONL file (e.g., `random_torch_v2_l1.jsonl`)
- `--level` (required): Generation level (0-4)
  - **Level 0**: Single operator per sample (47 operators total)
  - **Level 1**: Two operators per sample (all combinations: 47×47 = 2,209)
  - **Level 2**: Three operators per sample (randomly sampled)
  - **Level 3**: Five operators per sample (randomly sampled)
  - **Level 4**: Ten operators per sample (randomly sampled)
- `--target_count` (optional): Target number of samples to generate (default: 10000)

### Example Output Format

Each line in the output JSONL file contains:

```json
{
  "pytorch": "import torch\n\ndef get_inputs():\n    tensor_0 = torch.randn([32, 64], dtype=torch.float32)\n    ...\n\ndef fused_operator(tensor_0, tensor_1):\n    tensor_2 = torch.add(tensor_0, tensor_1)\n    return [tensor_2]",
  "difficulty": 1
}
```

## Architecture

### Core Components

1. **`ops.py`**: Defines tensor operations and their constraints
   - `BasicOp`: Base class for all operations
   - `Edge`: Represents tensor edges with shape constraints
   - Individual operation classes (Add, Matmul, ReLU, etc.)

2. **`main_cp_sat_v1.py`**: Graph generation using CP-SAT
   - `Graph`: Main graph class that generates computation graphs
   - Uses constraint programming to solve for valid tensor shapes
   - Ensures operations are mathematically valid

3. **`gen.py`**: Main generation script
   - Orchestrates multi-GPU parallel generation
   - Handles level-based operator selection
   - Validates generated code by executing it

### Generation Process

1. **Graph Construction**: Creates a computation graph with specified number of nodes
2. **Constraint Solving**: Uses CP-SAT solver to determine valid tensor shapes
3. **Code Generation**: Converts the solved graph to PyTorch code
4. **Validation**: Executes the generated code to ensure it runs correctly
5. **Output**: Writes validated samples to JSONL file

### Supported Operations

The generator supports 47+ operations organized by category:

**Elementwise Operations**:
- Add, Sub, Mul, Div, Maximum, Minimum, Lerp

**Linear Algebra**:
- Matmul, Bmm (Batch Matrix Multiply)

**Reductions**:
- Sum, Mean, Max, Min, ArgMax, ArgMin, Var

**Activations**:
- ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, LogSoftmax
- Swish, GELU, SELU, Hardsigmoid, Softplus, Softsign
- ELU, HardTanh, LogSigmoid

**Normalizations**:
- BatchNorm, LayerNorm, GroupNorm, InstanceNorm

**Convolutions & Pooling**:
- ConvNd (1D/2D/3D convolution)
- AvgPoolNd, MaxPoolNd

**Transforms**:
- Transpose, Cos, Sin, Exp2

**Tensor Operations**:
- Cat (Concatenate), Stack, CumSum, CumMax, CumMin

## Multi-GPU Generation

The generator automatically detects available GPUs and distributes work across them:

```bash
# With 4 GPUs, generating 10000 samples:
# GPU 0: samples 0-2499
# GPU 1: samples 2500-4999
# GPU 2: samples 5000-7499
# GPU 3: samples 7500-9999
```

Each GPU process:
- Generates samples independently
- Writes to a shared output file with locking
- Reports progress periodically

## Level Details

### Level 0: Single Operator
- **Nodes**: 1
- **Pattern**: One operator per sample, cycling through all 47 operators
- **Use Case**: Basic operator coverage, simple test cases

### Level 1: Two Operators
- **Nodes**: 2
- **Pattern**: All pairwise combinations (47×47 = 2,209 unique combinations)
- **Use Case**: Operator interaction testing

### Level 2: Three Operators
- **Nodes**: 3
- **Pattern**: Randomly sampled combinations (seed-based for reproducibility)
- **Use Case**: Moderate complexity

### Level 3: Five Operators
- **Nodes**: 5
- **Pattern**: Randomly sampled combinations
- **Use Case**: Higher complexity

### Level 4: Ten Operators
- **Nodes**: 10
- **Pattern**: Randomly sampled combinations
- **Use Case**: Maximum complexity, stress testing

## Configuration

### FLOPS Constraints

The generator uses FLOPS constraints to control computation complexity:

```python
# In gen.py, line 145:
src = get_sample(2**34, 2**35, num_nodes, op_types, level, gpu_id)
# min_flops = 2^34, max_flops = 2^35
```

### Tensor Shape Constraints

Defined in `ops.py`:
- Maximum dimensions: 5
- Maximum size per dimension: 8192
- Maximum tensor size: 2^30 elements
- Minimum tensor size: 2^5 elements

## Troubleshooting

### "Not feasible" Errors

If the CP-SAT solver cannot find a valid solution:
- The generator automatically reduces `min_flops` by half and retries
- If `min_flops` reaches 0, raises `ValueError`

## Integration with Training

The generated data can be used directly with verl training pipelines:

```python
from datasets import load_dataset

# Load generated data
dataset = load_dataset("json", data_files="random_torch_v2_l1.jsonl")

# Use in training
for item in dataset["train"]:
    pytorch_code = item["pytorch"]
    difficulty = item["difficulty"]
    # Train model to generate Triton kernels for pytorch_code
```

## Advanced Usage

### Custom Operator Selection

Modify `get_operator_combination_for_level()` in `gen.py` to customize operator selection per level.

### Custom FLOPS Range

Modify the `get_sample()` call in `worker_process()` to adjust FLOPS constraints:

```python
# More constrained (smaller range)
src = get_sample(2**33, 2**34, num_nodes, op_types, level, gpu_id)

# Less constrained (larger range)
src = get_sample(2**30, 2**36, num_nodes, op_types, level, gpu_id)
```

### Adding New Operations

1. Define the operation class in `ops.py` (inherit from `BasicOp`)
2. Implement `render()`, `flop()`, and shape constraints
3. Add to `DEFAULT_OP_SPECS` in `gen.py` and `main_cp_sat_v1.py`

## Examples

### Generate Level 1 dataset (10K samples)
```bash
python generator/gen.py \
    --output_file data/random_torch_v2_l1.jsonl \
    --level 1 \
    --target_count 10000
```

### Generate Level 0 dataset (all 47 operators)
```bash
python generator/gen.py \
    --output_file data/random_torch_v2_l0.jsonl \
    --level 0 \
    --target_count 47
```

### Generate Level 4 dataset (complex, 5K samples)
```bash
python generator/gen.py \
    --output_file data/random_torch_v2_l4.jsonl \
    --level 4 \
    --target_count 5000
```
