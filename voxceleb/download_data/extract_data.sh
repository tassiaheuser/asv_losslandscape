#!/bin/bash --login
#SBATCH --job-name=dataprep
#SBATCH --output=slurm_logs/%j_vc2prep.log
#SBATCH --error=slurm_logs/job_error_%j.log
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=fail

if command -v module >/dev/null 2>&1; then
    module load "${VOXCELEB_PYTHON_MODULE:-python/3.9.13-gpu}" || true
fi

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

python3 -m pip install -r requirements.txt

python3 download_data/dataprep.py --save_path data --extract
