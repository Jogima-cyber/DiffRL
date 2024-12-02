#!/bin/bash
#SBATCH --job-name=dmpc              # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=5:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=logs/logs_%j.out # output file name
#SBATCH --error=logs/logs_%j.err  # error file name
#SBATCH --array=1-20

#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60
# 20 30 40 50 60
# 20 30 40 50 60
# 20 30 40 50 60"
#     ["logdir"]="./logs/Ant/shac/20 ./logs/Ant/shac/30 ./logs/Ant/shac/40 ./logs/Ant/shac/50 ./logs/Ant/shac/60
# ./logs/Hopper/shac/20 ./logs/Hopper/shac/30 ./logs/Hopper/shac/40 ./logs/Hopper/shac/50 ./logs/Hopper/shac/60
# ./logs/Cheetah/shac/20 ./logs/Cheetah/shac/30 ./logs/Cheetah/shac/40 ./logs/Cheetah/shac/50 ./logs/Cheetah/shac/60
# ./logs/Humanoid/shac/20 ./logs/Humanoid/shac/30 ./logs/Humanoid/shac/40 ./logs/Humanoid/shac/50 ./logs/Humanoid/shac/60"
#     ["cfg"]="./cfg/shac/ant.yaml ./cfg/shac/ant.yaml ./cfg/shac/ant.yaml ./cfg/shac/ant.yaml ./cfg/shac/ant.yaml
# ./cfg/shac/hopper.yaml ./cfg/shac/hopper.yaml ./cfg/shac/hopper.yaml ./cfg/shac/hopper.yaml ./cfg/shac/hopper.yaml
# ./cfg/shac/cheetah.yaml ./cfg/shac/cheetah.yaml ./cfg/shac/cheetah.yaml ./cfg/shac/cheetah.yaml ./cfg/shac/cheetah.yaml
# ./cfg/shac/humanoid.yaml ./cfg/shac/humanoid.yaml ./cfg/shac/humanoid.yaml ./cfg/shac/humanoid.yaml ./cfg/shac/humanoid.yaml"
#)

#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60
# 20 30 40 50 60
# 20 30 40 50 60
# 20 30 40 50 60
# 20 30 40 50 60"
#     ["logdir"]="./logs/Ant/neuro_diff_sim/20 ./logs/Ant/neuro_diff_sim/30 ./logs/Ant/neuro_diff_sim/40 ./logs/Ant/neuro_diff_sim/50 ./logs/Ant/neuro_diff_sim/60
# ./logs/Hopper/neuro_diff_sim/20 ./logs/Hopper/neuro_diff_sim/30 ./logs/Hopper/neuro_diff_sim/40 ./logs/Hopper/neuro_diff_sim/50 ./logs/Hopper/neuro_diff_sim/60
# ./logs/Cheetah/neuro_diff_sim/20 ./logs/Cheetah/neuro_diff_sim/30 ./logs/Cheetah/neuro_diff_sim/40 ./logs/Cheetah/neuro_diff_sim/50 ./logs/Cheetah/neuro_diff_sim/60
# ./logs/Humanoid/neuro_diff_sim/20 ./logs/Humanoid/neuro_diff_sim/30 ./logs/Humanoid/neuro_diff_sim/40 ./logs/Humanoid/neuro_diff_sim/50 ./logs/Humanoid/neuro_diff_sim/60
# ./logs/SNUHumanoid/neuro_diff_sim/20 ./logs/SNUHumanoid/neuro_diff_sim/30 ./logs/SNUHumanoid/neuro_diff_sim/40 ./logs/SNUHumanoid/neuro_diff_sim/50 ./logs/SNUHumanoid/neuro_diff_sim/60"
#     ["cfg"]="./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml
# ./cfg/neuro_diff_sim/hopper.yaml ./cfg/neuro_diff_sim/hopper.yaml ./cfg/neuro_diff_sim/hopper.yaml ./cfg/neuro_diff_sim/hopper.yaml ./cfg/neuro_diff_sim/hopper.yaml
# ./cfg/neuro_diff_sim/cheetah.yaml ./cfg/neuro_diff_sim/cheetah.yaml ./cfg/neuro_diff_sim/cheetah.yaml ./cfg/neuro_diff_sim/cheetah.yaml ./cfg/neuro_diff_sim/cheetah.yaml
# ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml
# ./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml"
#)

#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60
# 20 30 40 50 60
# 20 30 40 50 60
# 20 30 40 50 60"
#     ["logdir"]="./logs/Ant/ppo/20 ./logs/Ant/ppo/30 ./logs/Ant/ppo/40 ./logs/Ant/ppo/50 ./logs/Ant/ppo/60
# ./logs/Hopper/ppo/20 ./logs/Hopper/ppo/30 ./logs/Hopper/ppo/40 ./logs/Hopper/ppo/50 ./logs/Hopper/ppo/60
# ./logs/Cheetah/ppo/20 ./logs/Cheetah/ppo/30 ./logs/Cheetah/ppo/40 ./logs/Cheetah/ppo/50 ./logs/Cheetah/ppo/60
# ./logs/Humanoid/ppo/20 ./logs/Humanoid/ppo/30 ./logs/Humanoid/ppo/40 ./logs/Humanoid/ppo/50 ./logs/Humanoid/ppo/60"
#     ["cfg"]="./cfg/ppo/ant.yaml ./cfg/ppo/ant.yaml ./cfg/ppo/ant.yaml ./cfg/ppo/ant.yaml ./cfg/ppo/ant.yaml
# ./cfg/ppo/hopper.yaml ./cfg/ppo/hopper.yaml ./cfg/ppo/hopper.yaml ./cfg/ppo/hopper.yaml ./cfg/ppo/hopper.yaml
# ./cfg/ppo/cheetah.yaml ./cfg/ppo/cheetah.yaml ./cfg/ppo/cheetah.yaml ./cfg/ppo/cheetah.yaml ./cfg/ppo/cheetah.yaml
# ./cfg/ppo/humanoid.yaml ./cfg/ppo/humanoid.yaml ./cfg/ppo/humanoid.yaml ./cfg/ppo/humanoid.yaml ./cfg/ppo/humanoid.yaml"
#)

#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60
# 20 30 40 50 60
# 20 30 40 50 60
# 20 30 40 50 60"
#     ["logdir"]="./logs/Ant/sac/20 ./logs/Ant/sac/30 ./logs/Ant/sac/40 ./logs/Ant/sac/50 ./logs/Ant/sac/60
# ./logs/Hopper/sac/20 ./logs/Hopper/sac/30 ./logs/Hopper/sac/40 ./logs/Hopper/sac/50 ./logs/Hopper/sac/60
# ./logs/Cheetah/sac/20 ./logs/Cheetah/sac/30 ./logs/Cheetah/sac/40 ./logs/Cheetah/sac/50 ./logs/Cheetah/sac/60
# ./logs/Humanoid/sac/20 ./logs/Humanoid/sac/30 ./logs/Humanoid/sac/40 ./logs/Humanoid/sac/50 ./logs/Humanoid/sac/60"
#     ["cfg"]="./cfg/sac/ant.yaml ./cfg/sac/ant.yaml ./cfg/sac/ant.yaml ./cfg/sac/ant.yaml ./cfg/sac/ant.yaml
# ./cfg/sac/hopper.yaml ./cfg/sac/hopper.yaml ./cfg/sac/hopper.yaml ./cfg/sac/hopper.yaml ./cfg/sac/hopper.yaml
# ./cfg/sac/cheetah.yaml ./cfg/sac/cheetah.yaml ./cfg/sac/cheetah.yaml ./cfg/sac/cheetah.yaml ./cfg/sac/cheetah.yaml
# ./cfg/sac/humanoid.yaml ./cfg/sac/humanoid.yaml ./cfg/sac/humanoid.yaml ./cfg/sac/humanoid.yaml ./cfg/sac/humanoid.yaml"
#)


#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60
# 20 30 40 50 60"
#     ["logdir"]="./logs/Ant/neuro_diff_sim/20 ./logs/Ant/neuro_diff_sim/30 ./logs/Ant/neuro_diff_sim/40 ./logs/Ant/neuro_diff_sim/50 ./logs/Ant/neuro_diff_sim/60
# ./logs/Humanoid/neuro_diff_sim/20 ./logs/Humanoid/neuro_diff_sim/30 ./logs/Humanoid/neuro_diff_sim/40 ./logs/Humanoid/neuro_diff_sim/50 ./logs/Humanoid/neuro_diff_sim/60"
#     ["cfg"]="./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml
# ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml"
#)
#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60"
#     ["logdir"]="./logs/Ant/neuro_diff_sim/20 ./logs/Ant/neuro_diff_sim/30 ./logs/Ant/neuro_diff_sim/40 ./logs/Ant/neuro_diff_sim/50 ./logs/Ant/neuro_diff_sim/60"
#     ["cfg"]="./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml"
#)

declare -A hyperparameters=(
     ["seed"]="20 30 40 50 60
 20 30 40 50 60
 20 30 40 50 60
 20 30 40 50 60"
     ["logdir"]="./logs/Hopper/neuro_diff_sim/20 ./logs/Hopper/neuro_diff_sim/30 ./logs/Hopper/neuro_diff_sim/40 ./logs/Hopper/neuro_diff_sim/50 ./logs/Hopper/neuro_diff_sim/60
 ./logs/Cheetah/neuro_diff_sim/20 ./logs/Cheetah/neuro_diff_sim/30 ./logs/Cheetah/neuro_diff_sim/40 ./logs/Cheetah/neuro_diff_sim/50 ./logs/Cheetah/neuro_diff_sim/60
 ./logs/Humanoid/neuro_diff_sim/20 ./logs/Humanoid/neuro_diff_sim/30 ./logs/Humanoid/neuro_diff_sim/40 ./logs/Humanoid/neuro_diff_sim/50 ./logs/Humanoid/neuro_diff_sim/60
 ./logs/SNUHumanoid/neuro_diff_sim/20 ./logs/SNUHumanoid/neuro_diff_sim/30 ./logs/SNUHumanoid/neuro_diff_sim/40 ./logs/SNUHumanoid/neuro_diff_sim/50 ./logs/SNUHumanoid/neuro_diff_sim/60"
     ["cfg"]="./cfg/neuro_diff_sim/hopper.yaml ./cfg/neuro_diff_sim/hopper.yaml ./cfg/neuro_diff_sim/hopper.yaml ./cfg/neuro_diff_sim/hopper.yaml ./cfg/neuro_diff_sim/hopper.yaml
 ./cfg/neuro_diff_sim/cheetah.yaml ./cfg/neuro_diff_sim/cheetah.yaml ./cfg/neuro_diff_sim/cheetah.yaml ./cfg/neuro_diff_sim/cheetah.yaml ./cfg/neuro_diff_sim/cheetah.yaml
 ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml
 ./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml"
)

#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60"
#     ["logdir"]="./logs/Humanoid/neuro_diff_sim/20 ./logs/Humanoid/neuro_diff_sim/30 ./logs/Humanoid/neuro_diff_sim/40 ./logs/Humanoid/neuro_diff_sim/50 ./logs/Humanoid/neuro_diff_sim/60"
#     ["cfg"]="./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml"
#)
#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60
# 20 30 40 50 60
# 20 30 40 50 60"
#     ["logdir"]="./logs/Ant/neuro_diff_sim/20 ./logs/Ant/neuro_diff_sim/30 ./logs/Ant/neuro_diff_sim/40 ./logs/Ant/neuro_diff_sim/50 ./logs/Ant/neuro_diff_sim/60
# ./logs/Hopper/neuro_diff_sim/20 ./logs/Hopper/neuro_diff_sim/30 ./logs/Hopper/neuro_diff_sim/40 ./logs/Hopper/neuro_diff_sim/50 ./logs/Hopper/neuro_diff_sim/60
# ./logs/Cheetah/neuro_diff_sim/20 ./logs/Cheetah/neuro_diff_sim/30 ./logs/Cheetah/neuro_diff_sim/40 ./logs/Cheetah/neuro_diff_sim/50 ./logs/Cheetah/neuro_diff_sim/60"
#     ["cfg"]="./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml ./cfg/neuro_diff_sim/ant.yaml
# ./cfg/neuro_diff_sim/hopper.yaml ./cfg/neuro_diff_sim/hopper.yaml ./cfg/neuro_diff_sim/hopper.yaml ./cfg/neuro_diff_sim/hopper.yaml ./cfg/neuro_diff_sim/hopper.yaml
# ./cfg/neuro_diff_sim/cheetah.yaml ./cfg/neuro_diff_sim/cheetah.yaml ./cfg/neuro_diff_sim/cheetah.yaml ./cfg/neuro_diff_sim/cheetah.yaml ./cfg/neuro_diff_sim/cheetah.yaml"
#)


#####
#####
#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60
# 20 30 40 50 60"
#     ["logdir"]="./logs/SNUHumanoid/neuro_diff_sim/20 ./logs/SNUHumanoid/neuro_diff_sim/30 ./logs/SNUHumanoid/neuro_diff_sim/40 ./logs/SNUHumanoid/neuro_diff_sim/50 ./logs/SNUHumanoid/neuro_diff_sim/60
# ./logs/Humanoid/neuro_diff_sim/20 ./logs/Humanoid/neuro_diff_sim/30 ./logs/Humanoid/neuro_diff_sim/40 ./logs/Humanoid/neuro_diff_sim/50 ./logs/Humanoid/neuro_diff_sim/60"
#     ["cfg"]="./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml
# ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml ./cfg/neuro_diff_sim/humanoid.yaml"
#)
#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60"
#     ["logdir"]="./logs/SNUHumanoid/neuro_diff_sim/20 ./logs/SNUHumanoid/neuro_diff_sim/30 ./logs/SNUHumanoid/neuro_diff_sim/40 ./logs/SNUHumanoid/neuro_diff_sim/50 ./logs/SNUHumanoid/neuro_diff_sim/60"
#     ["cfg"]="./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml ./cfg/neuro_diff_sim/snu_humanoid.yaml"
#)
#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60"
#     ["logdir"]="./logs/SNUHumanoid/shac/20 ./logs/SNUHumanoid/shac/30 ./logs/SNUHumanoid/shac/40 ./logs/SNUHumanoid/shac/50 ./logs/SNUHumanoid/shac/60"
#     ["cfg"]="./cfg/shac/snu_humanoid.yaml ./cfg/shac/snu_humanoid.yaml ./cfg/shac/snu_humanoid.yaml ./cfg/shac/snu_humanoid.yaml ./cfg/shac/snu_humanoid.yaml"
#)
#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60"
#     ["logdir"]="./logs/SNUHumanoid/ppo/20 ./logs/SNUHumanoid/ppo/30 ./logs/SNUHumanoid/ppo/40 ./logs/SNUHumanoid/ppo/50 ./logs/SNUHumanoid/ppo/60"
#     ["cfg"]="./cfg/ppo/snu_humanoid.yaml ./cfg/ppo/snu_humanoid.yaml ./cfg/ppo/snu_humanoid.yaml ./cfg/ppo/snu_humanoid.yaml ./cfg/ppo/snu_humanoid.yaml"
#)
#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60"
#     ["logdir"]="./logs/SNUHumanoid/sac/20 ./logs/SNUHumanoid/sac/30 ./logs/SNUHumanoid/sac/40 ./logs/SNUHumanoid/sac/50 ./logs/SNUHumanoid/sac/60"
#     ["cfg"]="./cfg/sac/snu_humanoid.yaml ./cfg/sac/snu_humanoid.yaml ./cfg/sac/snu_humanoid.yaml ./cfg/sac/snu_humanoid.yaml ./cfg/sac/snu_humanoid.yaml"
#)
#####
#####

#########
# Isaac #
#########
#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60"
#     ["logdir"]="./logs/Anymal/neuro_diff_sim/20 ./logs/Anymal/neuro_diff_sim/30 ./logs/Anymal/neuro_diff_sim/40 ./logs/Anymal/neuro_diff_sim/50 ./logs/Anymal/neuro_diff_sim/60"
#     ["cfg"]="./cfg/neuro_diff_sim/anymal.yaml ./cfg/neuro_diff_sim/anymal.yaml ./cfg/neuro_diff_sim/anymal.yaml ./cfg/neuro_diff_sim/anymal.yaml ./cfg/neuro_diff_sim/anymal.yaml"
#)
#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60"
#     ["logdir"]="./logs/Ingenuity/neuro_diff_sim/20 ./logs/Ingenuity/neuro_diff_sim/30 ./logs/Ingenuity/neuro_diff_sim/40 ./logs/Ingenuity/neuro_diff_sim/50 ./logs/Ingenuity/neuro_diff_sim/60"
#     ["cfg"]="./cfg/neuro_diff_sim/ingenuity.yaml ./cfg/neuro_diff_sim/ingenuity.yaml ./cfg/neuro_diff_sim/ingenuity.yaml ./cfg/neuro_diff_sim/ingenuity.yaml ./cfg/neuro_diff_sim/ingenuity.yaml"
#)
#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60"
#     ["logdir"]="./logs/Anymal/ppo/20 ./logs/Anymal/ppo/30 ./logs/Anymal/ppo/40 ./logs/Anymal/ppo/50 ./logs/Anymal/ppo/60"
#     ["cfg"]="./cfg/ppo/anymal.yaml ./cfg/ppo/anymal.yaml ./cfg/ppo/anymal.yaml ./cfg/ppo/anymal.yaml ./cfg/ppo/anymal.yaml"
#)
#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60"
#     ["logdir"]="./logs/Ingenuity/ppo/20 ./logs/Ingenuity/ppo/30 ./logs/Ingenuity/ppo/40 ./logs/Ingenuity/ppo/50 ./logs/Ingenuity/ppo/60"
#     ["cfg"]="./cfg/ppo/ingenuity.yaml ./cfg/ppo/ingenuity.yaml ./cfg/ppo/ingenuity.yaml ./cfg/ppo/ingenuity.yaml ./cfg/ppo/ingenuity.yaml"
#)
#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60"
#     ["logdir"]="./logs/BallBalance/ppo/20 ./logs/BallBalance/ppo/30 ./logs/BallBalance/ppo/40 ./logs/BallBalance/ppo/50 ./logs/BallBalance/ppo/60"
#     ["cfg"]="./cfg/ppo/ball_balance.yaml ./cfg/ppo/ball_balance.yaml ./cfg/ppo/ball_balance.yaml ./cfg/ppo/ball_balance.yaml ./cfg/ppo/ball_balance.yaml"
#)
#declare -A hyperparameters=(
#     ["seed"]="20 30 40 50 60"
#     ["logdir"]="./logs/BallBalance/neuro_diff_sim/20 ./logs/BallBalance/neuro_diff_sim/30 ./logs/BallBalance/neuro_diff_sim/40 ./logs/BallBalance/neuro_diff_sim/50 ./logs/BallBalance/neuro_diff_sim/60"
#     ["cfg"]="./cfg/neuro_diff_sim/ball_balance.yaml ./cfg/neuro_diff_sim/ball_balance.yaml ./cfg/neuro_diff_sim/ball_balance.yaml ./cfg/neuro_diff_sim/ball_balance.yaml ./cfg/neuro_diff_sim/ball_balance.yaml"
#)
#####
#####

hyperparam_args=""
for key in "${!hyperparameters[@]}"; do
    values=(${hyperparameters[$key]})
    num_values=${#values[@]}
    # Select the value based on the ID (assuming ID starts from 1)
    selected_value=${values[$(( (SLURM_ARRAY_TASK_ID - 1) % num_values ))]}
    # selected_value=${values[$((SLURM_ARRAY_TASK_ID - 1 >= num_values ? num_values - 1 : SLURM_ARRAY_TASK_ID - 1))]}
    hyperparam_args+=" --$key $selected_value"
done

cd ${SLURM_SUBMIT_DIR}

source ~/.bashrc
conda activate shac
#conda activate rlgpu
set -x
export WANDB_MODE=offline
cd examples

# Ant
#python train_neuro_diff_sim.py --cfg ./cfg/neuro_diff_sim/ant.yaml $hyperparam_args
#python train_shac.py --cfg ./cfg/shac/ant.yaml $hyperparam_args

# Hopper
#python train_neuro_diff_sim.py --cfg ./cfg/neuro_diff_sim/hopper.yaml $hyperparam_args
#python train_shac.py --cfg ./cfg/shac/hopper.yaml $hyperparam_args

# Cheetah
#python train_neuro_diff_sim.py --cfg ./cfg/neuro_diff_sim/cheetah.yaml $hyperparam_args
#python train_shac.py --cfg ./cfg/shac/cheetah.yaml $hyperparam_args

# Humanoid
#python train_neuro_diff_sim.py --cfg ./cfg/neuro_diff_sim/humanoid.yaml $hyperparam_args
#python train_shac.py --cfg ./cfg/shac/humanoid.yaml $hyperparam_args

#python train_shac.py --exp_name shac $hyperparam_args
#python train_neuro_diff_sim.py --exp_name neuro_diff_sim_2_rnn_valuehidden $hyperparam_args --env_type isaac_gym

# Dyna
#python train_neuro_diff_sim.py --exp_name neuro_diff_sim_2_dyna $hyperparam_args
python train_neuro_diff_sim.py --exp_name neuro_diff_sim_2_CONTROL_04nov $hyperparam_args
######

#python train_rl.py --exp_name ppo $hyperparam_args --env_type isaac_gym
#python train_rl.py --exp_name sac $hyperparam_args
#python train_pmo.py --exp_name pmo_512ac_3e_4me_lra1e4_fast_ISos_4cme_dyn_pred_batch_size1024_dynudt128_ent0 $hyperparam_args
# Abla
#python train_neuro_diff_sim.py --exp_name neuro_diff_sim_unroll_img_3 --env_type dflex --unroll-img $hyperparam_args

#IsaacGym
#python train_neuro_diff_sim.py --cfg ./cfg/neuro_diff_sim/shadow_hand.yaml --logdir ./logs/ShadowHand/neuro_diff_sim/20 --seed 20 --env_type isaac_gym --exp_name neuro_diff_sim_isaac_simnorm_1024ac_detach
#python train_neuro_diff_sim.py --cfg ./cfg/neuro_diff_sim/franka_cube_stack.yaml --logdir ./logs/FrankaCubeStack/neuro_diff_sim/20 --seed 20 --env_type isaac_gym --exp_name neuro_diff_sim_isaac_dynlr2e3
