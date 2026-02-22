import json
from pathlib import Path
from datasets import Dataset

TEMPLATE = Path("data/prompt_template/convert_pytorch_to_cuda_prompt.txt").read_text()

records = []
with open("processed_dataset/pytorch2cuda_sft_dataset_v4/gpt5_level0/gpt5_output.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if not r.get("is_pytorch_cuda_same") or r.get("compiler_error_msg"):
            continue  # skip 203 bad samples
        prompt = TEMPLATE.format(
            pytorch_code=r["pytorch_code"],
            pytorch_inputs=r["pytorch_inputs"],
        )
        records.append({
            "messages": [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": "```cpp\n" + r["cuda_code"] + "\n```"},
            ]
        })

Dataset.from_list(records).to_parquet("processed_dataset/pytorch2cuda_sft_dataset_v4/gpt5_level0/slime_sft.parquet")
print(f"Saved {len(records)} samples")  # expect 266