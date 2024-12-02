#!/bin/bash
#SBATCH --job-name=dmpc              # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=20           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=5:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/logs_%j.out # output file name
#SBATCH --error=logs/logs_%j.err  # error file name

cd ${SLURM_SUBMIT_DIR}

source ~/.bashrc
#conda activate shac
conda activate rlgpu
set -x
export WANDB_MODE=offline
cd examples

python train_neuro_diff_sim.py --cfg ./cfg/neuro_diff_sim/go2.yaml --logdir ./logs/Go2/neuro_diff_sim/20 --seed 20 --env_type isaac_gym --exp_name neuro_diff_sim_acc_5
