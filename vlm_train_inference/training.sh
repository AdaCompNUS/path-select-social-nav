nvidia-smi
conda activate llm

accelerate launch --config_file=./training/configs/deepspeed_zero{1,2,3}.yaml --num_processes 4 ./training/finetune_qwenvl25.py --dataset_name="threefruits/SCAND_path_selection" --model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct" --report_to="wandb" --learning_rate=3e-5 --per_device_train_batch_size=2 --gradient_accumulation_steps=1 --output_dir="./data/Qwen2.5-VL-path-selection" --logging_steps=5 --num_train_epochs=4 --gradient_checkpointing --remove_unused_columns=False --torch_dtype=float16 --fp16=True --push_to_hub=True