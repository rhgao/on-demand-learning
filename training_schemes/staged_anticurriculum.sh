#!/bin/bash
# This is the sample script for staged anti-curriculum learning

DATA_ROOT=dataset/my_train_set name=task level1=0 level2=0 level3=0 level4=0 level5=100 niter=200 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua

DATA_ROOT=dataset/my_train_set name=task netG=checkpoints/task_200_net_G.t7 base=200 level1=0 level2=0 level3=0 level4=100 level5=0 niter=200 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua

DATA_ROOT=dataset/my_train_set name=task netG=checkpoints/task_400_net_G.t7 base=400 level1=0 level2=0 level3=100 level4=0 level5=0 niter=200 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua

DATA_ROOT=dataset/my_train_set name=task netG=checkpoints/task_600_net_G.t7 base=600 level1=0 level2=100 level3=0 level4=0 level5=0 niter=200 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua

DATA_ROOT=dataset/my_train_set name=task netG=checkpoints/task_800_net_G.t7 base=800 level1=100 level2=0 level3=0 level4=0 level5=0 niter=200 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua
