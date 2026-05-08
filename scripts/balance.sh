#!/bin/bash

cd ../
export HYDRA_FULL_ERROR=1

# 2. 인자 받아오기
ALGO=$1        # 예: mappo, qmix
USE_WANDB=$2   # true or false
USE_RENDER=$3 # true or false
SEED_COUNT=$4  # 몇 번의 시드를 돌릴지 (예: 3이면 0,1,2)

# 3. 기본값 설정
if [ -z "$ALGO" ]; then ALGO="mappo"; fi
if [ -z "$USE_WANDB" ]; then USE_WANDB="false"; fi
if [ -z "$USE_RENDER" ]; then USE_RENDER="false"; fi
if [ -z "$SEED_COUNT" ]; then SEED_COUNT=1; fi

# 4. 하이퍼파라미터 설정 (Hydra 키워드에 맞춰 구성)
TASK=vmas/balance
LR=0.002
HIDDEN_DIM=256
CHECKPOINT_INTERVAL=0
MAX_STEPS=300  # 총 반복 횟수

# 5. 시드별 루프 실행
for ((i=0; i<$SEED_COUNT; i++))
do
    echo "Starting Experiment: $ALGO on $TASK with SEED $i"

    # BenchMARL 실행 (Hydra Override 문법 활용)
    python benchmarl/run.py \
        algorithm=$ALGO \
        task=$TASK \
        seed=$i \
        model.num_cells=[$HIDDEN_DIM,$HIDDEN_DIM] \
        experiment.lr=$LR \
        experiment.use_wandb=$USE_WANDB \
        experiment.use_render=$USE_RENDER\
        task.max_steps=$MAX_STEPS
done