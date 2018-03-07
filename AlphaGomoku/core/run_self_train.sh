#!/bin/bash
#SBATCH --time=6:00:00 #This is one hour
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=adrianalvarez15@gmail.com
#SBATCH --output=job-%j.log
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB#(This is max available RAM)
module load tensorflow/1.0.1-foss-2016a-Python-3.5.2-CUDA-7.5.18
module load tensorflow/1.2.0-foss-2016a-Python-3.5.2-CUDA-8.0.61
python3 run_gomoku_VS_terminal.py $*