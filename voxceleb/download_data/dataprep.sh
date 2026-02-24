#!/bin/bash --login
#SBATCH --job-name=dataprep_extract
#SBATCH --output=/home/users/t/tas.heu/masterthesis/logs/Voxceleb/dataprep_%j_256G.log
#SBATCH --error=/home/users/t/tas.heu/masterthesis/logs/Voxceleb/dataprep_%j_error.log
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=heuser.tassia@gmail.com
#SBATCH --partition=standard
#SBATCH --mem=256G
#~/masterthesis/Voxceleb_original/penv/bin/activate

echo "Starting job at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID at node: $SLURMD_NODENAME"

module load python/3.9.19

pip install tqdm
pip install -r /home/users/t/tas.heu/masterthesis/Voxceleb_original/requirements.txt

python3 ./dataprep.py  --save_path data --extract
