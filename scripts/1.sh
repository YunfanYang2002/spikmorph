#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python tools/train_ppo.py \
  --cfg configs/snn_stage_a.yaml \
  OUT_DIR output/snn_stage_a_pilot \
  ENV.WALKER_DIR output/unimals_100/train \
  DEVICE cuda:0 \
  MODEL.SPIKE.BACKEND torch \
  RNG_SEED 1409 \
  PPO.NUM_ENVS 32 \
  PPO.TIMESTEPS 2560 \
  PPO.BATCH_SIZE 1024 \
  PPO.MAX_STATE_ACTION_PAIRS 2000000

# 双 GPU 并行训练（取消下面各行开头的 # 后执行，或直接复制到终端）：
 CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
   --standalone \
   --nproc_per_node=3 \
   tools/train_ppo.py \
   --cfg configs/snn_stage_a.yaml \
   OUT_DIR output/snn_stage_a_parallel_10m \
   ENV.WALKER_DIR output/unimals_100/train \
   MODEL.SPIKE.BACKEND torch \
   RNG_SEED 1409 \
   PPO.NUM_ENVS 16 \
   PPO.TIMESTEPS 2560 \
   PPO.BATCH_SIZE 1024 \
   PPO.MAX_STATE_ACTION_PAIRS 10000000
