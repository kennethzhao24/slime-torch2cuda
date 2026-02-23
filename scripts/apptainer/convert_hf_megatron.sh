source scripts/models/qwen3-4B.sh

CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH=/root/Megatron-LM torchrun --nproc_per_node=4 \
  tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 2 \
  --hf-checkpoint /mnt/datasets/Qwen3-4B-Base \
  --make-vocab-size-divisible-by 1 \
  --save /mnt/datasets/qwen3_4b_base_torch_dist