
l=$1
OUTPUT_DIR=/u/yzhao25/slime-torch2cuda/torch2cuda/outputs/sft_datasets/gen
mkdir -p ${OUTPUT_DIR}
python gen.py \
   --output_file ${OUTPUT_DIR}/level${l}.jsonl \
   --level $l \
   --target_count 4096