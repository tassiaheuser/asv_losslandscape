#!/bin/bash --login
#SBATCH --job-name=dataprep
#SBATCH --output=/home/users/t/tas.heu/masterthesis/logs/dataprep/%j_vc2prep.log
#SBATCH --error=/home/users/t/tas.heu/masterthesis/logs/dataprep/job_error_%j.log
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=heuser.tassia@gmail.com

module load python/3.9.13-gpu

pip install tqdm

python3 ./dataprep.py --save_path data --extract
