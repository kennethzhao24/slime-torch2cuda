# Generating CUDA Kernels with Slime RL

**slime** is an LLM post-training framework for RL scaling, providing two core capabilities:

1.  **High-Performance Training**: Supports efficient training in various modes by connecting Megatron with SGLang;
2.  **Flexible Data Generation**: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

# TO-DO-List:
- [ ] Generator
  - [x] Dataset Prep
  - [x] Convert to slime Format
- [ ] GRPO
  - [x] [1N4G](scripts/run-qwen3-4B.sh)
  - [ ] 2N16G
- [ ] SFT

## Env Setup
Docker/appatiner is used for setup on [NCSA Delta](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/architecture.html).

### 1. Download the image from dockerhub and convert to apptainer format
```bash
apptainer pull slime.sif docker://slimerl/slime:latest
```
### 2. Request a GPU-interactive allocation, assh into the GPU, and run the container 
```bash
salloc --mem=220g --nodes=1 --ntasks-per-node=4 --cpus-per-task=4 --partition=gpuA100x4-interactive --account=bekz-delta-gpu --time=00:30:00 --gpus-per-node=4

ssh gpuaxxx

apptainer run --nv --bind /work/nvme/bekz/yzhao25/huggingface:/mnt/huggingface \ # bind huggingface cache path
                   --bind /work/nvme/bcrc/yzhao25/rl_datasets:/mnt/datasets \ # bind the datasets and model path
                   /u/yzhao25/slime/slime.sif \
                   /bin/bash --login
```

## Run Qwen3-4B with GRPO on DAPO-MATH

### 1. Download models and datasets
```bash
huggingface-cli download Qwen/Qwen3-4B --local-dir /work/nvme/bcrc/yzhao25/rl_datasets/Qwen3-4B

huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /work/nvme/bcrc/yzhao25/rl_datasets/dapo-math-17k

huggingface-cli download --repo-type dataset zhuzilin/aime-2024 --local-dir /work/nvme/bcrc/yzhao25/rl_datasets/aime-2024
```
### 2. Convert models to megatron format
```bash
source scripts/models/qwen3-4B.sh

CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=/root/Megatron-LM torchrun --nproc_per_node=4 \
  tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --tensor-model-parallel-size 2 \ # this should be consistent with your PERF_ARGS
  --pipeline-model-parallel-size 2 \
  --hf-checkpoint /mnt/datasets/Qwen3-4B \
  --make-vocab-size-divisible-by 1 \
  --save /mnt/datasets/qwen3_4b_torch_dist_tp2
```

### 3. Run GRPO on Qwen3-4B
```bash
bash scripts/run-qwen3-4B.sh 2>&1 | tee run.log
```

### 4. Alternatively, you can submit slurm jobs by running:
```bash
sbatch scripts/slurm_scripts/run_qwen3_4b.slurm
```


## Arguments Walkthrough

Arguments in slime are divided into three categories:

1.  **Megatron arguments**: slime reads all arguments in Megatron. You can configure Megatron by passing arguments like `--tensor-model-parallel-size 2`.
2.  **SGLang arguments**: All arguments for the installed SGLang are supported. These arguments must be prefixed with `--sglang-`. For example, `--mem-fraction-static` should be passed as `--sglang-mem-fraction-static`.
3.  **slime-specific arguments**: Please refer to: [slime/utils/arguments.py](slime/utils/arguments.py)

For complete usage instructions, please refer to the [Usage Documentation](docs/en/get_started/usage.md).


## FAQ & Acknowledgements

- For frequently asked questions, please see the [Q\&A](docs/en/get_started/qa.md)
- Special thanks to the following projects & communities: SGLang, Megatron‑LM, mbridge, OpenRLHF, veRL, Pai-Megatron-Patch and others.
- To quote slime, please use:

```bibtex
@misc{slime_github,
  author       = {Zilin Zhu and Chengxing Xie and Xin Lv and slime Contributors},
  title        = {slime: An LLM post-training framework for RL Scaling},
  year         = {2025},
  howpublished = {\url{https://github.com/THUDM/slime}},
  note         = {GitHub repository. Corresponding author: Xin Lv},
  urldate      = {2025-06-19}
}
```
