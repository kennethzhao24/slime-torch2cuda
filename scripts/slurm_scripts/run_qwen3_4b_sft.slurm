#!/bin/bash
#SBATCH --mem=220g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --time=24:00:00
#SBATCH --job-name=slime-rl-qwen3-4b-test
#SBATCH --account=bekz-delta-gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=/u/yzhao25/slime-torch2cuda/slurm-%j.out

apptainer run --nv --bind /work/nvme/bekz/yzhao25/huggingface:/mnt/huggingface \
                   --bind /work/nvme/bcrc/yzhao25/rl_datasets:/mnt/datasets \
                   /u/yzhao25/slime/slime.sif \
                   bash scripts/run-qwen3-4B-base-sft.sh 2>&1 | tee run_sft.log