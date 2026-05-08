#!/bin/bash

cd ../
export HYDRA_FULL_ERROR=1

ALGO=$1       
USE_WANDB=$2  
USE_RENDER=$3 
SEED_COUNT=$4

if [ -z "$ALGO" ]; then ALGO="mappo"; fi
if [ -z "$USE_WANDB" ]; then USE_WANDB="false"; fi
if [ -z "$USE_RENDER" ]; then USE_RENDER="false"; fi
if [ -z "$SEED_COUNT" ]; then SEED_COUNT=1; fi

TASK=vmas/football
LR=0.00005
HIDDEN_DIM=256
CHECKPOINT_INTERVAL=0
for ((i=0; i<$SEED_COUNT; i++))
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

        
done