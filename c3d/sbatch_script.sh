#!/bin/bash
#SBATCH --job-name TEST_C3D
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --account=hpeng1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAILNE
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
###SBATCH --mem-per-gpu=16g
#SBATCH --mem-per-cpu=3g
#SBATCH --get-user-env

### # SBATCH --cpus-per-task=1
# conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tp36dup
# CUDA_VISIBLE_DEVICES=0 
python c3d_test.py
# python bts_test_dataloader.py arguments_train_eigen_c3d.txt