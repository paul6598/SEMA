#!/bin/bash

cd ../
export HYDRA_FULL_ERROR=1
export PYGLET_HEADLESS=1
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0


ALGO=$1        # mappo, qmix
USE_WANDB=$2   # true or false
USE_RENDER=$3 # false(vllm과 충돌이 나 있어서 render는 false로 고정)
DESCRIPTION=$4 # 실험 설명 (예: "give_way_experiment")

# 3. 기본값 설정
if [ -z "$ALGO" ]; then ALGO="mappo"; fi
if [ -z "$USE_WANDB" ]; then USE_WANDB="false"; fi
if [ -z "$USE_RENDER" ]; then USE_RENDER="false"; fi
if [ -z "$DESCRIPTION" ]; then DESCRIPTION="default"; fi


TASK=vmas/give_way
LR=0.002
HIDDEN_DIM=256
CHECKPOINT_INTERVAL=0
MAX_N_FRAMES=4200000
MAX_STEPS=300
DONE_ON_COMPLETION=false
N_WORKERS=100
LLM_INTERVAL=50
INTENTION_VECTOR=4
SUBTASK_SIZE=5



EXTRA_ARGS=""
if [ "$ALGO" == "sema" ]; then
    EXTRA_ARGS="algorithm.LLM_interval=$LLM_INTERVAL \
                algorithm.intention_vector_size=$INTENTION_VECTOR \
fi

if [ "$ALGO" == "l2m2" ]; then
    EXTRA_ARGS="algorithm.LLM_interval=$LLM_INTERVAL \
                algorithm.subtask_size=$SUBTASK_SIZE"
fi

for i in 0
do 
    echo "Starting Experiment: $ALGO on $TASK with SEED $i"

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
        experiment.max_n_frames=$MAX_N_FRAMES \
        task.max_steps=$MAX_STEPS \
        task.done_on_completion=$DONE_ON_COMPLETION \
        $EXTRA_ARGS
done