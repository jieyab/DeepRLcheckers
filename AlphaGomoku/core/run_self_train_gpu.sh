#!/bin/bash
#SBATCH --time=6:00:00 #This is one hour
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=adrianalvarez15@gmail.com
#SBATCH --output=job-%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
module load tensorflow/1.0.1-foss-2016a-Python-3.5.2-CUDA-7.5.18
module load Boost
module load CUDA
module load tensorflow/1.2.0-foss-2016a-Python-3.5.2-CUDA-8.0.61
python3 run_gomoku_VS_terminal.py $*