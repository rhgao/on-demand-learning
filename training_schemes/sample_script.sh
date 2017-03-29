#!/bin/bash
# This is a sample script that can be adapted for different training schemes attempted in the paper, including on-demand learning, rigid-joint learning, staged (anti-)curriculum learning and cumulative (anti-)curriculum learning.

# The following code demonstrates staged-curriculum learning
DATA_ROOT=dataset/my_train_set name=task level1=100 level2=0 level3=0 level4=0 level5=0 niter=200 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua
DATA_ROOT=dataset/my_train_set name=task netG=checkpoints/task_200_net_G.t7 base=200 level1=0 level2=100 level3=0 level4=0 level5=0 niter=200 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua
DATA_ROOT=dataset/my_train_set name=task netG=checkpoints/task_400_net_G.t7 base=400 level1=0 level2=0 level3=100 level4=0 level5=0 niter=200 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua
DATA_ROOT=dataset/my_train_set name=task netG=checkpoints/task_600_net_G.t7 base=600 level1=0 level2=0 level3=0 level4=100 level5=0 niter=200 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua
DATA_ROOT=dataset/my_train_set name=task netG=checkpoints/task_800_net_G.t7 base=800 level1=0 level2=0 level3=0 level4=0 level5=100 niter=200 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua

# Rigid-Joint: level1=level2=level3=level4=level5=20
# Staged Curriculum: level1=100 -> level2=100 -> level3=100 -> level4=100 -> level5=100
# Staged Anti-Curriculum: level5=100 -> level4=100 -> level3=100 -> level2=100 -> level1=100
# Cumulative Curriculum: level1=100 -> level1=level2=50 -> level1=level2=level3=33 -> level1=level2=level3=level4=25 -> level1=level2=level3=level4=level5=20
# Cumulative Anti-Curriculum: level5=100 -> level4=level5=50 -> level3=level4=level5=33 -> level2=level3=level4=level5=25 -> level1=level2=level3=level4=level5=20
# On-Demand: validate on dataset/my_val_set to get the performance score for each sub-task (level1-5), assign the number of training examples per patch of each sub-task for the next epoch proportionally to its performance score
	# level1 = score1 * 100 / (score1  + score2 + score3 + score4 + score5)
	# level2 = score2 * 100 / (score1  + score2 + score3 + score4 + score5)
	# level3 = score3 * 100 / (score1  + score2 + score3 + score4 + score5)
	# level4 = score4 * 100 / (score1  + score2 + score3 + score4 + score5)
	# level5 = score5 * 100 / (score1  + score2 + score3 + score4 + score5)
