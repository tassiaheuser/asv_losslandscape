#!/bin/bash --login
#SBATCH --job-name=dataprep_extract
#SBATCH --output=slurm_logs/dataprep_%j_256G.log
#SBATCH --error=slurm_logs/dataprep_%j_error.log
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=fail
#SBATCH --partition=standard
#SBATCH --mem=256G
echo "Starting job at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID at node: $SLURMD_NODENAME"

if command -v module >/dev/null 2>&1; then
    module load "${VOXCELEB_PYTHON_MODULE:-python/3.9.19}" || true
fi

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

python3 -m pip install -r requirements.txt

python3 download_data/dataprep.py --save_path data --extract
