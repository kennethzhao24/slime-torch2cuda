import glob
import random
from pathlib import Path

from datasets import DatasetDict, DownloadConfig, load_dataset

def load_pytorch_cuda_sft_v2_as_trl_dataset(input_fn=None, num_proc=8):
    """
    Load <input_fn> as trl dataset.
    if <input_fn> not given, use  "processed_dataset/pytorch_cuda_sft_v2.jsonl".
    <input_fn> is a json line file. Each line is a json object string dump. For example:
    ```json
    {
        "pytorch_code": <pytorch_code_str>,
        "cuda_code": <cuda_code_str>,
        "input_tensor_shapes": <input_tensor_shapes_list>,
        "pytorch_inputs": <pytorch_inputs_str>,
    }
    ```
    
    Load this dataset as a huggingface dataset, where each example in the dataset is:
    ```
    {
        "prompt": <prompt_str>,
        "completion": <completion_str>,
        "raw_json_str": <the_original_json_str_from_jsonl>,
    }
    ```
    
    prompt_str is constructed by the following process:
    - load template from prompt_template/convert_pytorch_to_cuda_prompt.txt, denoted as template_str
    - prompt_str = template_str.format(pytorch_code=<pytorch_code_str>, input_tensor_shapes=<input_tensor_shapes_list>)
    """
    if input_fn is None:
        input_fn = "processed_dataset/pytorch2cuda_sft_dataset_v4/merged_level_0_1_positive_samples.jsonl"
    dataset_path = Path(input_fn)
    template_path = Path("/u/yzhao25/torch2cuda/data/prompt_template/convert_pytorch_to_cuda_prompt.txt")
    if num_proc is not None:
        if not isinstance(num_proc, int):
            raise TypeError(f"num_proc must be an int when provided, but received {type(num_proc).__name__}.")
        if num_proc < 1:
            raise ValueError("num_proc must be >= 1 when provided.")

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not template_path.is_file():
        raise FileNotFoundError(f"Prompt template file not found: {template_path}")

    template_str = template_path.read_text(encoding="utf-8")

    required_keys = ["pytorch_code", "cuda_code", "pytorch_inputs"]

    def preprocess_example(example):
        for key in required_keys:
            if key not in example:
                raise KeyError(f"Missing required key '{key}' in example.")

        pytorch_inputs = example["pytorch_inputs"]
        if not isinstance(pytorch_inputs, str):
            raise TypeError(
                f"Expected 'pytorch_inputs' to be 'str', but received {type(pytorch_inputs).__name__}."
            )
        prompt_str = template_str.format(
            pytorch_code=example["pytorch_code"],
            pytorch_inputs=pytorch_inputs,
        )
        
        completion = '```cpp\n' + example["cuda_code"] + '\n```'

        return {
            "prompt": prompt_str,
            "completion": completion,
        }

    raw_dataset = load_dataset(
        "json",
        data_files=str(dataset_path),
        split="train",
        download_config=DownloadConfig(local_files_only=True),
    )
    dataset = raw_dataset.map(
        preprocess_example,
        remove_columns=raw_dataset.column_names,
        num_proc=num_proc,
        desc="Formatting pytorch-cuda SFT dataset",
    )

    if len(dataset) == 0:
        raise ValueError(f"No records were loaded from {dataset_path}.")

    return DatasetDict({"train": dataset, "test": dataset}) # type: ignore


def load_stack_v2_cuda_as_trl_dataset(max_length=8192*2, streaming=False, num_proc=None) -> DatasetDict:
    PARQUET_PATTERN = "processed_dataset/stack_v2_cuda_parquet/*.parquet"
    parquet_files = sorted(glob.glob(PARQUET_PATTERN))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for pattern: {PARQUET_PATTERN}")
    if num_proc is not None:
        if streaming:
            raise ValueError("num_proc cannot be used when streaming=True.")
        if not isinstance(num_proc, int):
            raise TypeError(f"num_proc must be an int when provided, but received {type(num_proc).__name__}.")
        if num_proc < 1:
            raise ValueError("num_proc must be >= 1 when provided.")

    dataset = load_dataset(
        "parquet",
        data_files=parquet_files,
        split="train",
        download_config=DownloadConfig(local_files_only=True),
        streaming=streaming,
    )

    if "code_segment" not in dataset.column_names: # type: ignore
        raise ValueError("Stack v2 CUDA dataset is missing the 'code_segment' column.")

    def preprocess_function(example):
        example = dict(example)
        code_segment = example["code_segment"]
        if not isinstance(code_segment, str):
            raise TypeError(
                f"Expected 'code_segment' to be 'str', but received {type(code_segment).__name__}."
            )
        if len(code_segment) > max_length:
            max_start = len(code_segment) - max_length
            start_idx = random.randint(0, max_start)
            code_segment = code_segment[start_idx : start_idx + max_length]
        return {
            "text": code_segment,
        }

    dataset = dataset.map(preprocess_function, num_proc=num_proc)
    
    return DatasetDict({"train": dataset, "test": dataset}) # type: ignore

if __name__ == "__main__":
    load_pytorch_cuda_sft_v2_as_trl_dataset()
