#!/bin/bash
#$ -M kkosaraj@nd.edu
#$ -m abe
#$ -q gpu
#$ -l gpu_card=1
#$ -N DCBF_gamma=0.99_TS=3
#$ -o info
module load conda
module load cuda
module load cudnn
conda activate tf_gpu_krishna
python /afs/crc.nd.edu/user/k/kkosaraj/GITHUB_BUILD_1/microgrid_dcbf.py --gamma=0.99 --time_steps=3 --summary_dir=/afs/crc.nd.edu/user/k/kkosaraj/GITHUB_BUILD_1/my_scripts/test_name=dcbf_2/gamma=0.99/time_steps=3 > out.txt