#!/bin/bash
#SBATCH --time=4:00:00 #This is one hour
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=adrianalvarez15@gmail.com
#SBATCH --output=job-%j.log
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB#(This is max available RAM)
module load Python/3.5.2-foss-2016a
module load tensorflow/1.2.0-foss-2016a-Python-3.5.2
module load matplotlib/1.5.3-foss-2016a-Python-3.5.2
python3 pararell.py