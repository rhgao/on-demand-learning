#!/bin/bash
# This is the sample script for rigid joint learning

DATA_ROOT=dataset/my_train_set name=task level1=20 level2=20 level3=20 level4=20 level5=20 niter=1000 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua
