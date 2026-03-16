nvidia-smi
conda activate llm

vllm serve ./data/Qwen2.5-VL-path-selection --enforce-eager --max-model-len 32768 --quantization bitsandbytes --load-format bitsandbytes