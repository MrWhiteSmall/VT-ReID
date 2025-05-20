#!/bin/bash
#SBATCH --job-name=ccreid
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --ntasks=8
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gpu01
#SBATCH --mem=64GB
#SBATCH -o logs/job_%j.log

source activate ccreid

python main.py --dataset vcclothes -p 12902


