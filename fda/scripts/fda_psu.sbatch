#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s288057@studenti.polito.it
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 8
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --job-name=get_psu
#SBATCH --mem=2GB
##

module load nvidia/cudasdk/11.6
ml load intel/python/3/2019.4.088

source activate /home/ccuttano/.conda/envs/myenv/

python getSudoLabel_multi.py --cfg ./configs/SYNTHIAfda_psu.yml
