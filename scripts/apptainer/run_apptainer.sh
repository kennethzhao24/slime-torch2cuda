apptainer run --nv --bind /work/nvme/bekz/yzhao25/huggingface:/mnt/huggingface \
                   --bind /work/nvme/bcrc/yzhao25/rl_datasets:/mnt/datasets \
                   /u/yzhao25/slime/slime.sif \
                   /bin/bash --login

