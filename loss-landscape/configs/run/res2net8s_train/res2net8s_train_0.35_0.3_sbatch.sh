#!/bin/bash
####
#a) Define slurm job parameters
####

#SBATCH --job-name=8s_0.35_0.3  #6gIncRes

#resources:

#SBATCH --ntasks=1

##SBATCH --nodes=1

#SBATCH --cpus-per-task=8 # 14 is max for cpu-short (4 per 6 = 24)
# the job can use and see 4 CPUs (from max 24).
# needet task count -n, maybe there is a better way to specify cores

#SBATCH --mem=30G # Per CPU -> Per Core
##SBATCH --mem-per-cpu=200G # Per CPU -> Per Core
# the job will need 12GB of memory equally distributed on 4 cpus.(251GB are available in total on one node)

#SBATCH --gres=gpu:nvidia:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node) use SBATCH --gres=gpu:1080ti:1 to explicitly demand a Geforce 1080 Ti GPU. Use SBATCH --gres=gpu:A4000:1 to explicitly demand a RTX A4000 GPU

#SBATCH --error=slurm_logs/8s_0.35_0.3.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=slurm_logs/8s_0.35_0.3.%J.out
# write the standard output to job.*jobID*.out

####
#b) copy all needed data to the jobs scratch folder
# We copy the cifar10 datasets which is already available in common datasets folder to our job’s scratch folder.
# Note: For this script, cifar-10 sfno
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
####
if [ -n "${CONDA_EXE:-}" ]; then
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
    conda activate "${LOSS_LANDSCAPE_CONDA_ENV:-lola2}"
fi
# python plot_surface.py --config Output/resnext6g/23-05T11-14-20/config_23-05T11-14-20_resumeIncrease_HigherBatch.yml


# python plot_surface.py --config configs/vox2_v2_AAM.yaml
# python plot_surface.py --config configs/v2/vox2_v2_Speaker2.yaml

# python plot_surface.py --config configs/resnet.yaml

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
python plot_surface.py --config configs/run/res2net8s_train/res2net8s_train_0.35_0.3.yaml



# to kill a job go to the node where it runs and execute
# scontrol listpids
#  kill -INT <pid>

# squeue -o"%.7i %.9P %.8j %.8u %.2t %.10M %.6D %C"