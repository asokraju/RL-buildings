#!/bin/bash
#$ -M kkosaraj@nd.edu
#$ -m abe
#$ -q gpu
#$ -l gpu_card=1
#$ -N DCBF_gamma=0.9_TS=4
#$ -o info
module load conda
module load cuda
module load cudnn
conda activate tf_gpu_krishna
python /afs/crc.nd.edu/user/k/kkosaraj/GITHUB_BUILD_1/microgrid_dcbf.py --gamma=0.9 --time_steps=4 --summary_dir=/afs/crc.nd.edu/user/k/kkosaraj/GITHUB_BUILD_1/my_scripts/test_name=dcbf_1/gamma=0.9/time_steps=4 > out.txt
