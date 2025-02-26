# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import random

import numpy as np

import isaacgym
from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
from omegaconf import DictConfig, OmegaConf
import hydra
from utils.isaacgymenvs_make import isaacgym_task_map, make
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

import torch
import torch.nn as nn
import gym

import torch.nn.functional as F

'''
updates statistic from a full data
'''
class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RunningMeanStd, self).__init__()
        print('RunningMeanStd: ', insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0,2,3]
            if len(self.insize) == 2:
                self.axis = [0,2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0] 
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype = torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype = torch.float64))
        self.register_buffer("count", torch.ones((), dtype = torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, unnorm=False):
        if self.training:
            mean = input.mean(self.axis) # along channel axis
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count, 
                                                    mean, var, input.size()[0] )

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)
        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # get output


        if unnorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
        else:
            if self.norm_only:
                y = input/ torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # Actor MLP
        self.actor_mlp = nn.Sequential(
            nn.Linear(19, 256),
            nn.ELU(alpha=1.0),
            nn.Linear(256, 128),
            nn.ELU(alpha=1.0),
            nn.Linear(128, 64),
            nn.ELU(alpha=1.0)
        )

        # Critic MLP
        self.critic_mlp = nn.Sequential()  # Empty sequential block as described

        # Value head
        self.value = nn.Linear(64, 1)
        self.value_act = nn.Identity()

        # Mu (mean of the policy distribution)
        self.mu = nn.Linear(64, 7)
        self.mu_act = nn.Identity()

        self.sigma = nn.Parameter(torch.zeros(7))

        # Sigma (standard deviation of the policy distribution)
        self.sigma_act = nn.Identity()

        # Placeholder for CNN blocks (actor_cnn and critic_cnn are empty in this case)
        self.actor_cnn = nn.Sequential()
        self.critic_cnn = nn.Sequential()

    def forward(self, x):
        # Pass input through actor MLP
        x_actor = self.actor_mlp(x)

        # Compute value from value head
        value = self.value_act(self.value(x_actor))

        # Compute action distribution parameters (mu)
        mu = self.mu_act(self.mu(x_actor))

        # Sigma is undefined but the activation function is Identity (if required, extend here)
        return mu

class A2CNetwork(nn.Module):
    def __init__(self):
        super(A2CNetwork, self).__init__()
        self.a2c_network = Network()

    def forward(self, x):
        return self.a2c_network(x)

class ExtractObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return obs["obs"]

def generate_ppo(cfg: DictConfig, envs):
    checkpoint = cfg["checkpoint"]
    checkpoint_path = os.path.dirname(checkpoint)
    weights = torch.load(checkpoint)
    print(weights.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = A2CNetwork().to(device)
    actor.load_state_dict(weights["model"])
    actor.eval()

    obs_rms = RunningMeanStd(19).to(device)
    obs_rms.load_state_dict(weights["running_mean_std"])
    obs_rms.eval()

    envs = ExtractObsWrapper(envs)

    _init_cubeA_state = envs.env._init_cubeA_state.clone()
    _init_cubeB_state = envs.env._init_cubeB_state.clone()
    init_q = envs.env._q.clone()
    _qs = torch.zeros((envs.max_episode_length,) + init_q.shape, device=device)
    episode_lengths = torch.zeros(envs.num_envs, device=device)
    to_discard = torch.zeros(envs.num_envs, device=device)

    obs = envs.reset()

    for i in range(envs.max_episode_length):
        with torch.no_grad():
            actions = actor(envs.full2partial_state(obs_rms(obs)))

        next_obs, rewards, terminations, infos = envs.step(actions)
        obs = next_obs

        _qs[i] = envs.env._q.clone()

        episode_lengths += 1
        if(terminations.any()):
            done_idx = torch.argwhere(terminations).squeeze()
            to_discard[done_idx] = torch.logical_or(to_discard[done_idx], episode_lengths[done_idx] < envs.max_episode_length - 1).float()
            print(done_idx, episode_lengths[done_idx])

    print(to_discard.sum().item())
    file_path = os.path.join(checkpoint_path, 'saved_trajectories.pt')
    to_keep_ids = (1. - to_discard).nonzero(as_tuple = False).squeeze(-1)
    print(f"keeping {to_keep_ids.shape[0]} trajectories")
    tensor_dict = {
        '_init_cubeA_state': _init_cubeA_state[to_keep_ids],
        '_init_cubeB_state': _init_cubeB_state[to_keep_ids],
        'init_q': init_q[to_keep_ids],
        '_qs': _qs[:, to_keep_ids]
    }
    torch.save(tensor_dict, file_path)
    print(f"Tensors saved successfully at: {file_path}")

@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    if cfg.pbt.enabled:
        initial_pbt_check(cfg)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(
        cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True

    envs = make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg
        )

    if cfg.task_name == "FrankaCubeStack" and True:
        generate_ppo(cfg, envs)
    else:
        raise NotImplementedError("Generating trajectories is not available for this environment.")

if __name__ == "__main__":
    launch_rlg_hydra()
