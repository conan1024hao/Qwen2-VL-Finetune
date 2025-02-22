#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export MASTER_PORT=29501

export WANDB_RESUME="allow" &&
export WANDB_API_KEY="5b8322df11b04a8895325a5bf6ef8cbba0dd64a2" &&
export WANDB_ENTITY=conan1024hao &&
export WANDB_PROJECT=amasia &&
export WANDB_RUN_NAME="Qwen2.5-finetune-stage2-only" &&

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
GLOBAL_BATCH_SIZE=16
BATCH_PER_DEVICE=4
NUM_DEVICES=4
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

deepspeed --master_port $MASTER_PORT src/training/train.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /home/haowang/Amasia/visrecall_finetune_stage2.json \
    --image_folder /data/hao \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/$WANDB_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --min_pixels $((512 * 28 * 28)) \
    --max_pixels $((1280 * 28 * 28)) \
    --learning_rate 2e-6 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --dataloader_num_workers 0 \
    --report_to wandb \
    --run_name $WANDB_RUN_NAME

