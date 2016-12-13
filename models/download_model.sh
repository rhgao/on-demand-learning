#/bin/bash
TASK=$1
URL=vision.cs.utexas.edu/projects/on_demand_learning/models/${TASK}_net_G.t7
MODEL_FILE=${TASK}_net_G.t7
wget -N $URL -O $MODEL_FILE
