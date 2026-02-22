# Torch2CUDA


## 1. Generate differnt levels of PyTorch Code
```bash
cd generator
sh run_generator.sh ${level} # choose your own level
```


## 2. Generate validation set

```bash
ray start --head 

python run_ray_gen_pytorch_code.py --level 2 --target_count 1024 --output_file processed_dataset/pytorch2cuda_test_set/level_2/pytorch_script_to_rollout.jsonl
```
