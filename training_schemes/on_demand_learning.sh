#!/bin/bash
# This is the sample script for the on-demand learning training scheme. Please adapt it for your usage based on your own desired validation implementation. The idea is to validate regularly to assess the performance of the snapshot model. The assessment can be based on different factors of your own design, like the average reconstruction loss, average PSNR (currently used in the arXiv paper), etc. The optimal way to get the performance scores for each sub-tasks will be released in the future after publication at a conference.

# Train for one epoch to get an initial model
DATA_ROOT=dataset/my_train_set name=task level1=20 level2=20 level3=20 level4=20 level5=20 niter=1 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua

COUNTER=1
while [ $COUNTER -lt 1000 ]; do
	#validate on dataset/my_val_set to get the performance score for each sub-task (level1-5)
	#assign the number of training examples per patch of each sub-task for the next epoch proportionally to its performance score
	# num_level1 = score1 * 100 / (score1  + score2 + score3 + score4 + score5)
	# num_level2 = score2 * 100 / (score1  + score2 + score3 + score4 + score5)
	# num_level3 = score3 * 100 / (score1  + score2 + score3 + score4 + score5)
	# num_level4 = score4 * 100 / (score1  + score2 + score3 + score4 + score5)
	# num_level5 = score5 * 100 / (score1  + score2 + score3 + score4 + score5)
	DATA_ROOT=dataset/my_train_set name=task netG=checkpoints/task_${COUNTER}_net_G.t7 base=$COUNTER level1=num_level1 level2=num_level2 level3=num_level3 level4=num_level4 level5=num_level5 niter=1 loadSize=96 fineSize=64 display=1 display_iter=50 gpu=1 th train.lua
	let COUNTER=COUNTER+1
done
