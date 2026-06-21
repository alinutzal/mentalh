module load vllm/0.14.1

hf cache scan
# 2 GPUs (salloc -C gpu -t 1:00:00 -c 4 -A pls0144 --gpus=2)
vllm serve Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 3200 \
    --tensor-parallel-size 2 \
    --disable_custom_all_reduce


# vllm serve Qwen/Qwen3-30B-A3B \



