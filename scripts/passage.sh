#!/bin/bash

cd ../
export HYDRA_FULL_ERROR=1

# 2. 인자 받아오기
ALGO=$1        # 예: mappo, qmix
USE_WANDB=$2   # true or false
USE_RENDER=$3 # true or false
DESCRIPTION=$4 # 실험 설명 (예: "give_way_experiment")

# 3. 기본값 설정
if [ -z "$ALGO" ]; then ALGO="mappo"; fi
if [ -z "$USE_WANDB" ]; then USE_WANDB="false"; fi
if [ -z "$USE_RENDER" ]; then USE_RENDER="false"; fi
if [ -z "$DESCRIPTION" ]; then DESCRIPTION="default"; fi

# 4. 하이퍼파라미터 설정 (Hydra 키워드에 맞춰 구성)
TASK=vmas/passage
LR=0.002
HIDDEN_DIM=256
CHECKPOINT_INTERVAL=0
MAX_STEPS=500
DONE_ON_COMPLETION=false
N_WORKERS=100
LLM_INTERVAL=50
LLM_INTERVAL_DECAY=false
INTENTION_VECTOR=4
SUBTASK_SIZE=5
EXTRA_ARGS=""
if [ "$ALGO" == "larlpassage" ]; then
    EXTRA_ARGS="algorithm.LLM_interval=$LLM_INTERVAL \
                algorithm.LLM_interval_decay=$LLM_INTERVAL_DECAY \
                algorithm.intention_vector_size=$INTENTION_VECTOR"
fi
if [ "$ALGO" == "l2m2" ]; then
    EXTRA_ARGS="algorithm.LLM_interval=$LLM_INTERVAL \
                algorithm.LLM_interval_decay=$LLM_INTERVAL_DECAY \
                algorithm.subtask_size=$SUBTASK_SIZE"
fi

# 5. 시드별 루프 실행
for i in 10 11 12 13 14
do
    echo "Starting Experiment: $ALGO on $TASK with SEED $i"

    # BenchMARL 실행 (Hydra Override 문법 활용)
    python benchmarl/run.py \
        algorithm=$ALGO \
        task=$TASK \
        seed=$i \
        model.num_cells=[$HIDDEN_DIM,$HIDDEN_DIM] \
        experiment.lr=$LR \
        experiment.checkpoint_interval=$CHECKPOINT_INTERVAL \
        experiment.use_wandb=$USE_WANDB \
        experiment.use_render=$USE_RENDER \
        experiment.on_policy_n_envs_per_worker=$N_WORKERS \
        experiment.description=$DESCRIPTION \
        task.max_steps=$MAX_STEPS \
        $EXTRA_ARGS

done