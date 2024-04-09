#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --output=bayes_tuning.out
export PATH=/vol/bitbucket/mg2720/fypvenv/bin/:$PATH
source activate
which python
source /vol/cuda/12.2.0/setup.sh
cd /vol/bitbucket/mg2720/dgbl-coursework/GSR-Net
python grid_search.py
