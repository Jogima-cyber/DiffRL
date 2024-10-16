# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from multiprocessing.sharedctypes import Value
import sys, os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)

import time
import numpy as np
import copy
from tensorboardX import SummaryWriter
import yaml

from utils.common import *
import torch
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
import utils.torch_utils as tu
from utils.running_mean_std import RunningMeanStd
from utils.dataset import CriticDataset, CriticAdvDataset
from utils.time_report import TimeReport
from utils.average_meter import AverageMeter
import models.actor
import models.critic
import models.dyn_model

from typing import NamedTuple
import gym
from gym.spaces import Box

class ReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor

class ReplayBuffer():
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size,
        observation_space,
        action_space,
        storing_device = "cpu",
        training_device = "cpu",
        n_envs = 1,
        optimize_memory_usage = False,
        handle_timeout_termination = True,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = observation_space.shape

        self.action_dim = int(np.prod(action_space.shape))
        self.pos = 0
        self.full = False
        self.storing_device = storing_device
        self.training_device = training_device
        self.n_envs = n_envs

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = torch.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=torch.float32, device = storing_device)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = torch.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=torch.float32, device = storing_device)

        self.actions = torch.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=torch.float32, device = storing_device)

        self.rewards = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device = storing_device)
        self.dones = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device = storing_device)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device = storing_device)

    def add(
        self,
        obs,
        next_obs,
        action,
        reward,
        done,
        truncateds = None
    ):
        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = obs.clone().to(self.storing_device)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = next_obs.clone().to(self.storing_device)

        self.actions[self.pos] = action.clone().to(self.storing_device)
        self.rewards[self.pos] = reward.clone().to(self.storing_device)
        self.dones[self.pos] = done.clone().to(self.storing_device)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = truncateds.to(self.storing_device)

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size, env = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            upper_bound = self.buffer_size if self.full else self.pos
            batch_inds = np.random.randint(0, upper_bound, size=batch_size)
            return self._get_samples(batch_inds, env=env)

        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds, env = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :].to(self.training_device),
            self.actions[batch_inds, env_indices, :].to(self.training_device),
            next_obs.to(self.training_device),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1).to(self.training_device),
            self.rewards[batch_inds, env_indices].reshape(-1, 1).to(self.training_device),
        )
        return ReplayBufferSamples(*data)

# Taken from https://github.com/nicklashansen/tdmpc2/blob/5f6fadec0fec78304b4b53e8171d348b58cac486/tdmpc2/common/math.py#L5C1-L9C47
def soft_ce(pred, target, num_bins, vmin, vmax):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, num_bins, vmin, vmax)
    return -(target * pred).sum(-1, keepdim=True)

def two_hot(x, num_bins, vmin, vmax):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if num_bins == 0:
        return x
    elif num_bins == 1:
        return symlog(x)
    bin_size = (vmax - vmin) / (num_bins - 1)
    x = torch.clamp(symlog(x), vmin, vmax).squeeze(1)
    bin_idx = torch.floor((x - vmin) / bin_size).long()
    bin_offset = ((x - vmin) / bin_size - bin_idx.float()).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.size(0), num_bins, device=x.device)
    soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % num_bins, bin_offset)
    return soft_two_hot

def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))

class NeuroDiffSim:
    def __init__(self, cfg):
        self.env_type = cfg["params"]["general"]["env_type"]

        seeding(cfg["params"]["general"]["seed"])

        if self.env_type == "dflex":
            import dflex as df
            import envs
            env_fn = getattr(envs, cfg["params"]["diff_env"]["name"])
            self.env = env_fn(num_envs = cfg["params"]["config"]["num_actors"], \
                              device = cfg["params"]["general"]["device"], \
                              render = cfg["params"]["general"]["render"], \
                              seed = cfg["params"]["general"]["seed"], \
                              episode_length=cfg["params"]["diff_env"].get("episode_length", 250), \
                              stochastic_init = cfg["params"]["diff_env"].get("stochastic_env", True), \
                              MM_caching_frequency = cfg["params"]['diff_env'].get('MM_caching_frequency', 1), \
                              no_grad = True)
            self.max_episode_length = self.env.episode_length
        elif self.env_type == "isaac_gym":
            import isaacgym
            import isaacgymenvs
            self.env = isaacgymenvs.make(
                seed=cfg["params"]["general"]["seed"],
                task=cfg["params"]["isaac_gym"]["name"],
                num_envs=cfg["params"]["config"]["num_actors"],
                sim_device="cuda:0",
                rl_device="cuda:0",
                headless=True
            )
            self.env.num_actions = self.env.num_acts
            self.max_episode_length = self.env.max_episode_length
            self.num_dyn_obs = self.env.num_dyn_obs
        else:
            raise ValueError(
                f"env type {self.env_type} is not supported."
            )

        print('num_envs = ', self.env.num_envs)
        print('num_actions = ', self.env.num_actions)
        print('num_obs = ', self.env.num_obs)

        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions

        self.device = cfg["params"]["general"]["device"]

        self.gamma = cfg['params']['config'].get('gamma', 0.99)

        self.critic_method = cfg['params']['config'].get('critic_method', 'one-step') # ['one-step', 'td-lambda']
        if self.critic_method == 'td-lambda':
            self.lam = cfg['params']['config'].get('lambda', 0.95)

        self.steps_num = cfg["params"]["config"]["steps_num"]
        self.max_epochs = cfg["params"]["config"]["max_epochs"]
        self.actor_lr = float(cfg["params"]["config"]["actor_learning_rate"])
        self.critic_lr = float(cfg['params']['config']['critic_learning_rate'])
        self.dyn_model_lr = float(cfg['params']['config']['dyn_model_learning_rate'])
        self.lr_schedule = cfg['params']['config'].get('lr_schedule', 'linear')
        
        self.target_critic_alpha = cfg['params']['config'].get('target_critic_alpha', 0.4)

        self.obs_rms = None
        if cfg['params']['config'].get('obs_rms', False):
            if self.env_type == "dflex":
                self.obs_rms = RunningMeanStd(shape = (self.num_obs), device = self.device)
            else:
                self.obs_rms = RunningMeanStd(shape = (self.num_dyn_obs), device = self.device)
            
        self.ret_rms = None
        if cfg['params']['config'].get('ret_rms', False):
            self.ret_rms = RunningMeanStd(shape = (), device = self.device)

        self.rew_scale = cfg['params']['config'].get('rew_scale', 1.0)

        self.critic_iterations = cfg['params']['config'].get('critic_iterations', 16)
        self.num_batch = cfg['params']['config'].get('num_batch', 4)
        self.batch_size = self.num_envs * self.steps_num // self.num_batch
        self.name = cfg['params']['config'].get('name', "Ant")

        self.truncate_grad = cfg["params"]["config"]["truncate_grads"]
        self.grad_norm = cfg["params"]["config"]["grad_norm"]
        
        if cfg['params']['general']['train']:
            self.log_dir = cfg["params"]["general"]["logdir"]
            os.makedirs(self.log_dir, exist_ok = True)
            # save config
            save_cfg = copy.deepcopy(cfg)
            if 'general' in save_cfg['params']:
                deleted_keys = []
                for key in save_cfg['params']['general'].keys():
                    if key in save_cfg['params']['config']:
                        deleted_keys.append(key)
                for key in deleted_keys:
                    del save_cfg['params']['general'][key]

            if cfg['params']['config'].get('wandb_track', False):
                import wandb

                cfg['env_id'] = cfg['params']['diff_env']['name']
                cfg['exp_name'] = cfg["params"]["general"]['exp_name']

                wandb.init(
                    project=cfg['params']['config']['wandb_project_name'],
                    sync_tensorboard=True,
                    config=cfg,
                    name=self.name,
                    monitor_gym=True,
                    save_code=True,
                )

            yaml.dump(save_cfg, open(os.path.join(self.log_dir, 'cfg.yaml'), 'w'))
            self.writer = SummaryWriter(os.path.join(self.log_dir, 'log'))
            # save interval
            self.save_interval = cfg["params"]["config"].get("save_interval", 500)
            # stochastic inference
            self.stochastic_evaluation = True
        else:
            self.stochastic_evaluation = not (cfg['params']['config']['player'].get('determenistic', False) or cfg['params']['config']['player'].get('deterministic', False))
            self.steps_num = self.env.episode_length

        # create actor critic network
        self.actor_name = cfg["params"]["network"].get("actor", 'ActorStochasticMLP') # choices: ['ActorDeterministicMLP', 'ActorStochasticMLP']
        self.critic_name = cfg["params"]["network"].get("critic", 'CriticMLP')

        self.multi_modal_cor = cfg['params']['general']['multi_modal_cor']
        if self.multi_modal_cor:
            self.dyn_model_name = "StochSSMCor"
        else:
            self.dyn_model_name = cfg["params"]["network"].get("dyn_model", 'StochSSM')
        actor_fn = getattr(models.actor, self.actor_name)
        self.actor = actor_fn(self.num_obs, self.num_actions, cfg['params']['network'], device = self.device)
        critic_fn = getattr(models.critic, self.critic_name)
        if self.env_type == "dflex":
            self.critic = critic_fn(self.num_obs, cfg['params']['network'], device = self.device)
        else:
            self.critic = critic_fn(self.num_dyn_obs, cfg['params']['network'], device = self.device) # /!\ or partial obs ?

        ########################
        self.avantage_objective = cfg['params']['config'].get('avantage_objective', False)
        if self.avantage_objective:
            self.critic_adv_name = cfg["params"]["network"].get("critic_adv", 'CriticAdvMLP')
            critic_adv_fn = getattr(models.critic, self.critic_adv_name)
            if self.env_type == "dflex":
                self.critic_adv = critic_adv_fn(self.num_obs, self.num_actions, cfg['params']['network'], device = self.device)
            else:
                self.critic_adv = critic_adv_fn(self.num_dyn_obs, self.num_actions, cfg['params']['network'], device = self.device) # /!\ or partial obs ?
        ########################

        self.learn_reward = cfg["params"]["config"].get("learn_reward", False)
        if self.learn_reward:
            self.num_bins = cfg['params']['network']['dyn_model_mlp']['num_bins']
            self.vmin = cfg['params']['network']['dyn_model_mlp']['vmin']
            self.vmax = cfg['params']['network']['dyn_model_mlp']['vmax']
        dyn_model_fn = getattr(models.dyn_model, self.dyn_model_name)
        if self.env_type == "dflex":
            self.dyn_model = dyn_model_fn(self.num_obs, self.num_actions, cfg['params']['network'], self.learn_reward, device = self.device)
        else:
            self.dyn_model = dyn_model_fn(self.num_dyn_obs, self.num_actions, cfg['params']['network'], self.learn_reward, device = self.device)
        self.all_params = list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.dyn_model.parameters())
        self.target_critic = copy.deepcopy(self.critic)

        if cfg['params']['general']['train']:
            self.save('init_policy')

        # initialize optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), betas = cfg['params']['config']['betas'], lr = self.actor_lr)
        if self.avantage_objective:
            self.critic_optimizer = torch.optim.Adam(list(self.critic.parameters()) + list(self.critic_adv.parameters()), betas = cfg['params']['config']['betas'], lr = self.critic_lr)
        else:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), betas = cfg['params']['config']['betas'], lr = self.critic_lr)
        self.dyn_model_optimizer = torch.optim.Adam(self.dyn_model.parameters(), lr = self.dyn_model_lr)

        # replay buffer
        if self.env_type == "dflex":
            self.obs_buf = torch.zeros((self.steps_num, self.num_envs, self.num_obs), dtype = torch.float32, device = self.device)
        else:
            self.obs_buf = torch.zeros((self.steps_num, self.num_envs, self.num_dyn_obs), dtype = torch.float32, device = self.device)
        self.rew_buf = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.done_mask = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.next_values = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.target_values = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
        self.ret = torch.zeros((self.num_envs), dtype = torch.float32, device = self.device)
        if self.avantage_objective:
            self.values = torch.zeros((self.steps_num + 1, self.num_envs), dtype = torch.float32, device = self.device)
            self.advantages = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)
            self.act_buf = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
            self.done_buf = torch.zeros((self.steps_num, self.num_envs), dtype = torch.float32, device = self.device)

        # real replay buffer for dyn model
        self.dyn_buffer_size = int(float(cfg["params"]["config"].get("dyn_buffer_size", 5e6)))
        if self.env_type == "dflex":
            self.dyn_rb = ReplayBuffer(
                self.dyn_buffer_size,
                Box(low=np.NINF,high=np.PINF,shape=np.array([self.num_obs])),
                Box(low=np.NINF,high=np.PINF,shape=np.array([self.num_actions])),
                storing_device = "cpu",
                training_device = self.device,
                n_envs = self.num_envs,
            )
        else:
            self.dyn_rb = ReplayBuffer(
                self.dyn_buffer_size,
                Box(low=np.NINF,high=np.PINF,shape=np.array([self.num_dyn_obs])),
                Box(low=np.NINF,high=np.PINF,shape=np.array([self.num_actions])),
                storing_device = "cpu",
                training_device = self.device,
                n_envs = self.num_envs,
            )

        self.dyn_udt = int(cfg["params"]["config"].get("dyn_udt", 256))
        self.init_dyn_udt = int(cfg["params"]["config"].get("init_dyn_udt", 1000))
        self.min_replay = int(cfg["params"]["config"].get("min_replay", 4000))
        self.dyn_pred_batch_size = int(cfg["params"]["config"].get("dyn_pred_batch_size", 1024))
        self.pretrained = False
        self.filter_sigma_events = cfg["params"]["config"].get("filter_sigma_events", False)
        self.unroll_img = cfg["params"]["general"]['unroll_img']

        # for kl divergence computing
        self.old_mus = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.old_sigmas = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.mus = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)
        self.sigmas = torch.zeros((self.steps_num, self.num_envs, self.num_actions), dtype = torch.float32, device = self.device)

        # counting variables
        self.iter_count = 0
        self.step_count = 0

        # loss variables
        self.episode_length_his = []
        self.episode_loss_his = []
        self.episode_discounted_loss_his = []
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype = int, device = self.device)
        self.best_policy_loss = np.inf
        self.actor_loss = np.inf
        self.value_loss = np.inf

        # average meter
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)

        # timer
        self.time_report = TimeReport()

    def fill_replay_buffer(self, deterministic = False):
        filled_nb = 0
        with torch.no_grad():
            if self.env_type == "dflex":
                obs = self.env.initialize_trajectory()
            else:
                obs = self.env.dyn_obs_buf.clone()
            raw_obs = obs.clone()
            if self.obs_rms is not None:
                # update obs rms
                with torch.no_grad():
                    self.obs_rms.update(obs)
                # normalize the current obs
                obs = self.obs_rms.normalize(obs)
            while filled_nb < self.min_replay:
                if self.env_type == "dflex":
                    actions = self.actor(obs, deterministic = deterministic)
                    next_obs, rew, done, extra_info = self.env.neurodiff_step(torch.tanh(actions))
                    real_next_obs = next_obs.clone()
                    if done.any():
                        done_idx = torch.argwhere(done).squeeze()
                        real_next_obs[done_idx] = extra_info['obs_before_reset'][done_idx]
                else:
                    actions = self.actor(self.env.full2partial_state(obs), deterministic = deterministic)
                    next_obs, rew, done, extra_info = self.env.step(torch.tanh(actions))
                    done = extra_info['dones']
                    next_obs = self.env.dyn_obs_buf.clone()
                    real_next_obs = next_obs.clone()
                    if done.any():
                        done_idx = torch.argwhere(done).squeeze()
                        real_next_obs[done_idx] = extra_info['obs_before_reset'][done_idx]

                self.dyn_rb.add(raw_obs.detach(), real_next_obs.detach(), actions.detach(), rew.detach(), done.float(), done.float())
                filled_nb += self.num_envs
                raw_obs = next_obs.clone()
                if self.obs_rms is not None:
                    # update obs rms
                    with torch.no_grad():
                        self.obs_rms.update(next_obs.clone())
                    # normalize the current obs
                    obs = self.obs_rms.normalize(next_obs.clone())
        print("Filling done")

    def compute_actor_loss(self, deterministic = False):
        rew_acc = torch.zeros((self.steps_num + 1, self.num_envs), dtype = torch.float32, device = self.device)
        gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        next_values = torch.zeros((self.steps_num + 1, self.num_envs), dtype = torch.float32, device = self.device)

        actor_loss = torch.tensor(0., dtype = torch.float32, device = self.device)
        #dyn_loss = torch.tensor(0., dtype = torch.float32, device = self.device) ###

        with torch.no_grad():
            if self.obs_rms is not None:
                obs_rms = copy.deepcopy(self.obs_rms)

            if self.ret_rms is not None:
                ret_var = self.ret_rms.var.clone()

        # initialize trajectory to cut off gradients between episodes.
        if self.env_type == "dflex":
            obs = self.env.initialize_trajectory()
        else:
            obs = self.env.dyn_obs_buf.clone()
        raw_obs = obs.clone()
        #raw_obs = obs.clone().requires_grad_(True) ###
        if self.unroll_img:
            true_raw_obs = obs.clone()
        if self.obs_rms is not None:
            # update obs rms
            with torch.no_grad():
                if self.unroll_img:
                    self.obs_rms.update(true_raw_obs)
                else:
                    self.obs_rms.update(obs)
            # normalize the current obs
            obs = obs_rms.normalize(obs)
        #samples_used = 0. ##
        #active_trajectories = torch.ones(self.num_envs, dtype=torch.bool, device = self.device) ##
        for i in range(self.steps_num):
            # collect data for critic training
            with torch.no_grad():
                self.obs_buf[i] = obs.clone()
                if self.avantage_objective:
                    if self.env_type == "dflex":
                        self.values[i] = self.target_critic(obs).squeeze(-1)
                    else:
                        self.values[i] = self.target_critic(self.env.full2partial_state(obs.clone())).squeeze(-1)

            if self.unroll_img:
                #actions = self.actor(obs.detach(), deterministic = deterministic)
                if self.env_type == "dflex":
                    actions = self.actor(obs, deterministic = deterministic)
                else:
                    actions = self.actor(self.env.full2partial_state(obs.clone()), deterministic = deterministic)
            else:
                if self.env_type == "dflex":
                    actions = self.actor(obs, deterministic = deterministic)
                else:
                    actions = self.actor(self.env.full2partial_state(obs.clone()), deterministic = deterministic)

            with torch.no_grad():
                if self.avantage_objective:
                    self.act_buf[i] =  torch.tanh(actions.clone())

            #obs, rew, done, extra_info = self.env.step(torch.tanh(actions))
            #if((~active_trajectories).any()): ##
            #    inactive_idx = torch.argwhere(~active_trajectories).squeeze() ##
            #    actions[inactive_idx] = actions[inactive_idx].detach() ##

            if self.unroll_img:
                real_next_obs = self.dyn_model(obs, torch.tanh(actions))[0] + raw_obs
                if self.env_type == "dflex":
                    next_obs, rew, done, extra_info = self.env.neurodiff_step(torch.tanh(actions.detach()))
                else:
                    next_obs, rew, done, extra_info = self.env.step(torch.tanh(actions.detach()))
                    next_obs = self.env.dyn_obs_buf.clone()
            else:
                #with torch.no_grad():
                #    img_real_next_obs = self.dyn_model(obs, torch.tanh(actions))[0] + raw_obs ##
                if self.multi_modal_cor:
                    real_next_obs, next_obs, rew, done, extra_info = models.dyn_model.DynamicsFunctionCor.apply(raw_obs, actions, self.dyn_model, self.env, obs_rms)
                else:
                    real_next_obs, next_obs, rew, done, extra_info = models.dyn_model.DynamicsFunction.apply(raw_obs, actions, self.dyn_model, self.env, obs_rms, self.env_type)
            replay_addable = True

            #with torch.no_grad():
            #    alpha = 1 ##
            #    bad_gradients_idx = ((real_next_obs - img_real_next_obs).abs() > 2 * alpha * torch.sqrt(obs_rms.var + 1e-5).unsqueeze(0)).any(-1) ##
            #    samples_used += active_trajectories.sum() ##
            #    active_trajectories = torch.logical_and(active_trajectories, ~bad_gradients_idx) ##

            if self.filter_sigma_events and self.env_type == "dflex":
                if (~torch.isfinite(next_obs)).sum() > 0:
                    print_warning("Got inf next_obs from sim")
                    nan_idx = torch.any(~torch.isfinite(next_obs), dim=-1)
                    next_obs[nan_idx] = 0.0
                    replay_addable = False

                if (~torch.isfinite(real_next_obs)).sum() > 0:
                    print_warning("Got inf real_next_obs from sim")
                    nan_idx = torch.any(~torch.isfinite(real_next_obs), dim=-1)
                    real_next_obs[nan_idx] = 0.0
                    replay_addable = False

                nan_idx = torch.any(real_next_obs.abs() > 1e6, dim=-1)
                if nan_idx.sum() > 0:
                    print_warning("Got large real_next_obs from sim")
                    real_next_obs[nan_idx] = 0.0
                    replay_addable = False

                nan_idx = torch.any(next_obs.abs() > 1e6, dim=-1)
                if nan_idx.sum() > 0:
                    print_warning("Got large next_obs from sim")
                    next_obs[nan_idx] = 0.0
                    replay_addable = False

                alpha = 6
                below_alphasigmas_masks = torch.any(real_next_obs < (-alpha * torch.sqrt(obs_rms.var + 1e-5) + obs_rms.mean).unsqueeze(0), dim=-1)
                above_alphasigmas_masks = torch.any(real_next_obs > (alpha * torch.sqrt(obs_rms.var + 1e-5) + obs_rms.mean).unsqueeze(0), dim=-1)
                alphasigmas_masks = torch.logical_or(below_alphasigmas_masks, above_alphasigmas_masks)
                if alphasigmas_masks.any():
                    print_warning("Got real_next_obs not in alpha sigma interval from sim")
                    replay_addable = False

            if replay_addable:
                if self.unroll_img:
                    true_real_next_obs = next_obs.clone()
                    if(done.any()):
                        done_idx = torch.argwhere(done).squeeze()
                        true_real_next_obs[done_idx] = extra_info['obs_before_reset'][done_idx]
                    self.dyn_rb.add(true_raw_obs.detach(), true_real_next_obs.detach(), actions.detach(), rew.detach(), done.float(), done.float())
                else:
                    self.dyn_rb.add(raw_obs.detach(), real_next_obs.detach(), actions.detach(), rew.detach(), done.float(), done.float())

            ##### an ugly fix for simulation nan values #### # reference: https://github.com/pytorch/pytorch/issues/15131
            if self.filter_sigma_events:
                def create_hook():
                    def hook(grad):
                        torch.nan_to_num(grad, 0.0, 0.0, 0.0, out = grad)
                    return hook

                if real_next_obs.requires_grad:
                    real_next_obs.register_hook(create_hook())
                if actions.requires_grad:
                    actions.register_hook(create_hook())
            #################################################

            with torch.no_grad():
                raw_rew = rew.clone()

            # Differential reward recomputation
            if self.learn_reward:
                recalculated_rew = models.dyn_model.RewardsFunction.apply(raw_obs, actions, rew.clone(), self.dyn_model, obs_rms)
            else:
                #real_next_obs = models.dyn_model.GradientAnalysorFunction.apply(real_next_obs.clone())
                recalculated_rew = self.env.diffRecalculateReward(real_next_obs, torch.tanh(actions))
                #if (rew != recalculated_rew).any() and not self.unroll_img:
                if not torch.allclose(rew, recalculated_rew, rtol=1e-05, atol=1e-08, equal_nan=False) and not self.unroll_img and self.env_type == "dflex":
                    print(i, (rew != recalculated_rew), rew, recalculated_rew, (rew - recalculated_rew))
                    print('recalculated reward error')
                    raise ValueError
                rew = recalculated_rew.clone()
            #print(rew.mean())

            #second_diff_actions = actions.detach().clone()
            #second_diff_actions.requires_grad_(True)
            #second_diff_real_next_obs = models.dyn_model.DynamicsFunctionPMO.apply(raw_obs.detach(), second_diff_actions, real_next_obs.detach(), self.dyn_model, obs_rms)
            #second_diff_rew = self.env.diffRecalculateReward(second_diff_real_next_obs, torch.tanh(second_diff_actions))
            #gradients = torch.autograd.grad(outputs=second_diff_rew, inputs=second_diff_actions, ###
            #                   grad_outputs=torch.ones_like(second_diff_rew), ###
            #                   create_graph=True, retain_graph=True)[0] ###

            #second_diff_obs = raw_obs.detach().clone()
            #second_diff_obs.requires_grad_(True)
            #second_diff_actions = actions.detach().clone()
            #second_diff_actions.requires_grad_(True)
            #second_diff_real_next_obs = models.dyn_model.DynamicsFunctionPMO.apply(second_diff_obs, second_diff_actions, real_next_obs.detach(), self.dyn_model, obs_rms)
            #second_diff_rew = self.env.diffRecalculateReward(second_diff_real_next_obs, torch.tanh(second_diff_actions))
            #gradients = torch.autograd.grad(outputs=second_diff_rew, inputs=second_diff_obs, ###
            #                   grad_outputs=torch.ones_like(second_diff_rew), ###
            #                   create_graph=True, retain_graph=True)[0] ###

            #gradients = torch.autograd.grad(outputs=rew, inputs=actions, ###
            #                   grad_outputs=torch.ones_like(rew), ###
            #                   create_graph=True, retain_graph=True)[0] ###
            #gradients_norm = torch.sum(gradients ** 2, dim=1) ###
            #dyn_loss += gradients_norm.sum() ###
            ###################################

            # Next obs recomputation, in case of reset, we must put back the state after reset into obs
            prev_obs = obs.clone()
            obs = real_next_obs.clone()
            if done.any():
                done_idx = torch.argwhere(done).squeeze()
                obs[done_idx] = next_obs[done_idx].detach() # Cut off from the graph
            ########################

            #with torch.no_grad():
            #    raw_rew = rew.clone()

            # scale the reward
            rew = rew * self.rew_scale

            raw_obs = obs.clone()
            if self.unroll_img:
                true_raw_obs = next_obs.clone()
            if self.obs_rms is not None:
                # update obs rms
                with torch.no_grad():
                    if self.unroll_img:
                        self.obs_rms.update(true_raw_obs)
                    else:
                        self.obs_rms.update(obs)
                # normalize the current obs
                obs = obs_rms.normalize(obs)

            if self.ret_rms is not None:
                # update ret rms
                with torch.no_grad():
                    self.ret = self.ret * self.gamma + rew
                    self.ret_rms.update(self.ret)

                rew = rew / torch.sqrt(ret_var + 1e-6)

            self.episode_length += 1

            done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)

            if self.env_type == "dflex":
                next_values[i + 1] = self.target_critic(obs).squeeze(-1)
            else:
                next_values[i + 1] = self.target_critic(self.env.full2partial_state(obs.clone())).squeeze(-1)

            for id in done_env_ids:
                if self.env_type == "dflex" and (torch.isnan(extra_info['obs_before_reset'][id]).sum() > 0 \
                    or torch.isinf(extra_info['obs_before_reset'][id]).sum() > 0 \
                    or (torch.abs(extra_info['obs_before_reset'][id]) > 1e6).sum() > 0): # ugly fix for nan values
                    next_values[i + 1, id] = 0.
                elif self.episode_length[id] < self.max_episode_length: # early termination
                    next_values[i + 1, id] = 0.
                else: # otherwise, use terminal value critic to estimate the long-term performance
                    if self.obs_rms is not None:
                        real_obs = obs_rms.normalize(real_next_obs[id])
                    else:
                        real_obs = real_next_obs[id]
                    if self.env_type == "dflex":
                        next_values[i + 1, id] = self.target_critic(real_obs).squeeze(-1)
                    else:
                        next_values[i + 1, id] = self.target_critic(self.env.full2partial_state(real_obs.clone())).squeeze(-1)
            
            if (next_values[i + 1] > 1e6).sum() > 0 or (next_values[i + 1] < -1e6).sum() > 0:
                print('next value error')
                raise ValueError
            
            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            if i < self.steps_num - 1:
                #to_cut_env_ids = torch.logical_or(done, bad_gradients_idx).nonzero(as_tuple = False).squeeze(-1) ##
                ##actor_loss = actor_loss + (- rew_acc[i + 1, done_env_ids] - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]).sum()
                #actor_loss = actor_loss + (- rew_acc[i + 1, to_cut_env_ids] - self.gamma * gamma[to_cut_env_ids] * next_values[i + 1, to_cut_env_ids]).sum() ##
                if self.avantage_objective:
                    actor_loss = actor_loss + (- self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]).sum()
                else:
                    actor_loss = actor_loss + (- rew_acc[i + 1, done_env_ids] - self.gamma * gamma[done_env_ids] * next_values[i + 1, done_env_ids]).sum()
            else:
                # terminate all envs at the end of optimization iteration
                ##actor_loss = actor_loss + (- rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]).sum()
                if self.avantage_objective:
                    actor_loss = actor_loss + (- self.gamma * gamma * next_values[i + 1, :]).sum()
                else:
                    actor_loss = actor_loss + (- rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]).sum()

            if self.avantage_objective:
                actor_loss = actor_loss - self.critic_adv(prev_obs, torch.tanh(actions)).sum()

            #actor_loss = actor_loss - next_values[i + 1, :].sum()

            # compute gamma for next step
            gamma = gamma * self.gamma

            # clear up gamma and rew_acc for done envs
            gamma[done_env_ids] = 1.
            rew_acc[i + 1, done_env_ids] = 0.

            # collect data for critic training
            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.steps_num - 1:
                    self.done_mask[i] = done.clone().to(torch.float32)
                else:
                    self.done_mask[i, :] = 1.
                self.next_values[i] = next_values[i + 1].clone()
                if self.avantage_objective:
                    self.done_buf[i] = done.clone().to(torch.float32)

            # collect episode loss
            with torch.no_grad():
                self.episode_loss -= raw_rew
                self.episode_discounted_loss -= self.episode_gamma * raw_rew
                self.episode_gamma *= self.gamma
                if len(done_env_ids) > 0:
                    self.episode_loss_meter.update(self.episode_loss[done_env_ids])
                    self.episode_discounted_loss_meter.update(self.episode_discounted_loss[done_env_ids])
                    self.episode_length_meter.update(self.episode_length[done_env_ids])
                    for done_env_id in done_env_ids:
                        if (self.episode_loss[done_env_id] > 1e6 or self.episode_loss[done_env_id] < -1e6):
                            print('ep loss error')
                            raise ValueError

                        self.episode_loss_his.append(self.episode_loss[done_env_id].item())
                        self.episode_discounted_loss_his.append(self.episode_discounted_loss[done_env_id].item())
                        self.episode_length_his.append(self.episode_length[done_env_id].item())
                        self.episode_loss[done_env_id] = 0.
                        self.episode_discounted_loss[done_env_id] = 0.
                        self.episode_length[done_env_id] = 0
                        self.episode_gamma[done_env_id] = 1.

        actor_loss /= self.steps_num * self.num_envs * self.steps_num / 2

        #dyn_loss /= self.steps_num * self.num_envs ###
        #actor_loss /= samples_used ##
        #self.discarded_ratio = 1 - samples_used / (self.steps_num * self.num_envs) ##

        if self.ret_rms is not None:
            actor_loss = actor_loss * torch.sqrt(ret_var + 1e-6)

        self.actor_loss = actor_loss.detach().cpu().item()
        #self.dyn_loss = dyn_loss.detach().cpu().item() ###

        self.step_count += self.steps_num * self.num_envs

        if self.avantage_objective:
            with torch.no_grad():
                if self.env_type == "dflex":
                    self.values[-1] = self.target_critic(obs).squeeze(-1)
                else:
                    self.values[-1] = self.target_critic(self.env.full2partial_state(obs.clone())).squeeze(-1)

        #self.dyn_model_optimizer.zero_grad()
        #(dyn_loss * 2).backward(retain_graph = True) # 2 for action on ant
        #self.dyn_model_optimizer.step()

        return actor_loss

    @torch.no_grad()
    def evaluate_policy(self, num_games, deterministic = False):
        episode_length_his = []
        episode_loss_his = []
        episode_discounted_loss_his = []
        episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        episode_length = torch.zeros(self.num_envs, dtype = int)
        episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)

        obs = self.env.reset()

        games_cnt = 0
        while games_cnt < num_games:
            if self.obs_rms is not None:
                obs = self.obs_rms.normalize(obs)

            actions = self.actor(obs, deterministic = deterministic)

            obs, rew, done, _ = self.env.step(torch.tanh(actions))

            episode_length += 1

            done_env_ids = done.nonzero(as_tuple = False).squeeze(-1)

            episode_loss -= rew
            episode_discounted_loss -= episode_gamma * rew
            episode_gamma *= self.gamma
            if len(done_env_ids) > 0:
                for done_env_id in done_env_ids:
                    print('loss = {:.2f}, len = {}'.format(episode_loss[done_env_id].item(), episode_length[done_env_id]))
                    episode_loss_his.append(episode_loss[done_env_id].item())
                    episode_discounted_loss_his.append(episode_discounted_loss[done_env_id].item())
                    episode_length_his.append(episode_length[done_env_id].item())
                    episode_loss[done_env_id] = 0.
                    episode_discounted_loss[done_env_id] = 0.
                    episode_length[done_env_id] = 0
                    episode_gamma[done_env_id] = 1.
                    games_cnt += 1
        
        mean_episode_length = np.mean(np.array(episode_length_his))
        mean_policy_loss = np.mean(np.array(episode_loss_his))
        mean_policy_discounted_loss = np.mean(np.array(episode_discounted_loss_his))
 
        return mean_policy_loss, mean_policy_discounted_loss, mean_episode_length

    @torch.no_grad()
    def compute_target_values(self):
        if self.avantage_objective:
            lastgaelam = 0
            for t in reversed(range(self.steps_num)):
                nextnonterminal = 1.0 - self.done_buf[t]
                nextvalues = self.values[t + 1]
                delta = self.rew_buf[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                self.advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        if self.critic_method == 'one-step':
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == 'td-lambda':
            Ai = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
            Bi = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
            lam = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
            for i in reversed(range(self.steps_num)):
                lam = lam * self.lam * (1. - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (self.lam * self.gamma * Ai + self.gamma * self.next_values[i] + (1. - lam) / (1. - self.lam) * self.rew_buf[i])
                Bi = self.gamma * (self.next_values[i] * self.done_mask[i] + Bi * (1.0 - self.done_mask[i])) + self.rew_buf[i]
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError

    def compute_critic_loss(self, batch_sample):
        predicted_values = self.critic(batch_sample['obs']).squeeze(-1)
        target_values = batch_sample['target_values']
        critic_loss = ((predicted_values - target_values) ** 2).mean()

        if self.avantage_objective:
            predicted_advantages = self.critic_adv(batch_sample['obs'], batch_sample['act']).squeeze(-1)
            target_advantages = batch_sample['target_advantages']
            critic_adv_loss = ((predicted_advantages - target_advantages) ** 2).mean()
            return critic_loss + critic_adv_loss

        return critic_loss

    def initialize_env(self):
        if self.env_type == "dflex":
            self.env.clear_grad()
            self.env.reset()
        #else:
        #    self.env.launch()

    @torch.no_grad()
    def run(self, num_games):
        mean_policy_loss, mean_policy_discounted_loss, mean_episode_length = self.evaluate_policy(num_games = num_games, deterministic = not self.stochastic_evaluation)
        print_info('mean episode loss = {}, mean discounted loss = {}, mean episode length = {}'.format(mean_policy_loss, mean_policy_discounted_loss, mean_episode_length))

    def train_dyn_model(self):
        train_steps = self.dyn_udt
        if not self.pretrained:
            train_steps = self.init_dyn_udt
            print("pre training")

        log_total_dyn_loss = 0.0
        log_total_dyn_stoch_loss = 0.0
        log_total_reward_loss = 0.0

        if self.pretrained:
            data = self.dyn_rb.sample(self.dyn_pred_batch_size * train_steps)
            perm_indices = torch.randperm(self.dyn_pred_batch_size * train_steps)
        for ts in range(train_steps):
            if not self.pretrained:
                data = self.dyn_rb.sample(self.dyn_pred_batch_size)
                observations_concat = data.observations
                actions_concat = data.actions
                next_observations_concat = data.next_observations
                if self.learn_reward:
                    reward_concat = data.rewards
            else:
                cur_perm_indices = perm_indices[ts * self.dyn_pred_batch_size : (ts + 1) * self.dyn_pred_batch_size]
                observations_concat = data.observations[cur_perm_indices]
                actions_concat = data.actions[cur_perm_indices]
                next_observations_concat = data.next_observations[cur_perm_indices]
                if self.learn_reward:
                    reward_concat = data.rewards[cur_perm_indices]

            if self.obs_rms is not None:
                # normalize the current obs
                state = self.obs_rms.normalize(observations_concat.clone())
                if self.multi_modal_cor:
                    next_state = self.obs_rms.normalize(next_observations_concat.clone())

            if self.multi_modal_cor:
                latent = self.dyn_model.encode(state, torch.tanh(actions_concat), next_state)
                pred = self.dyn_model(state, torch.tanh(actions_concat), latent)
            else:
                pred = self.dyn_model(state, torch.tanh(actions_concat))
            pred_next_logvar = pred[1]
            pred_next_obs = pred[0]
            if self.learn_reward:
                pred_reward = self.dyn_model.reward(state, torch.tanh(actions_concat))

            target_observations = next_observations_concat - observations_concat

            inv_var = torch.exp(-pred_next_logvar)
            mse_loss_inv = (torch.pow(pred_next_obs - target_observations, 2) * inv_var).mean()
            var_loss = pred_next_logvar.mean()
            loss = mse_loss_inv + var_loss
            if self.learn_reward:
                if self.num_bins == 0:
                    reward_loss = F.mse_loss(pred_reward, reward_concat)
                    log_total_reward_loss += reward_loss.item()
                    loss += reward_loss
                else:
                    reward_loss = soft_ce(pred_reward, reward_concat, self.num_bins, self.vmin, self.vmax).mean()
                    log_total_reward_loss += reward_loss.item()
                    loss += reward_loss

            self.dyn_model_optimizer.zero_grad()
            loss.backward()
            self.dyn_model_optimizer.step()

            with torch.no_grad():
                log_total_dyn_stoch_loss += loss.item()

                loss = F.mse_loss(pred_next_obs, target_observations, reduction = "none").mean()
                log_total_dyn_loss += loss.item()

        self.writer.add_scalar("scalars/dyn_loss", log_total_dyn_loss / ts, self.iter_count)
        self.writer.add_scalar("scalars/reward_loss", log_total_reward_loss / ts, self.iter_count)
        self.writer.add_scalar("scalars/dyn_stoch_loss", log_total_dyn_stoch_loss / ts, self.iter_count)
        self.dyn_loss = log_total_dyn_loss / ts
        if not self.pretrained:
            print("pre training end")
            self.pretrained = True

    def train(self):
        self.start_time = time.time()

        # add timers
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("compute actor loss")
        self.time_report.add_timer("forward simulation")
        self.time_report.add_timer("backward simulation")
        self.time_report.add_timer("prepare critic dataset")
        self.time_report.add_timer("actor training")
        self.time_report.add_timer("critic training")

        self.time_report.start_timer("algorithm")

        # initializations
        self.initialize_env()
        self.episode_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_discounted_loss = torch.zeros(self.num_envs, dtype = torch.float32, device = self.device)
        self.episode_length = torch.zeros(self.num_envs, dtype = int, device = self.device)
        self.episode_gamma = torch.ones(self.num_envs, dtype = torch.float32, device = self.device)
        
        def actor_closure():
            #self.actor_optimizer.zero_grad()
            self.time_report.start_timer("compute actor loss")

            self.time_report.start_timer("forward simulation")
            actor_loss = self.compute_actor_loss()
            self.time_report.end_timer("forward simulation")

            self.time_report.start_timer("backward simulation")
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.time_report.end_timer("backward simulation")

            with torch.no_grad():
                self.grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
                if self.truncate_grad:
                    clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.grad_norm_after_clip = tu.grad_norm(self.actor.parameters())
                
                # sanity check
                if torch.isnan(self.grad_norm_before_clip): #or self.grad_norm_before_clip > 1000000.:
                    print('NaN gradient')
                    raise ValueError

            self.time_report.end_timer("compute actor loss")

            return actor_loss

        self.fill_replay_buffer()

        # main training process
        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()

            # learning rate schedule
            if self.lr_schedule == 'linear':
                actor_lr = (1e-5 - self.actor_lr) * float(epoch / self.max_epochs) + self.actor_lr
                for param_group in self.actor_optimizer.param_groups:
                    param_group['lr'] = actor_lr
                lr = actor_lr
                critic_lr = (1e-5 - self.critic_lr) * float(epoch / self.max_epochs) + self.critic_lr
                for param_group in self.critic_optimizer.param_groups:
                    param_group['lr'] = critic_lr
            else:
                lr = self.actor_lr

            # dyn model training
            self.train_dyn_model()

            # train actor
            self.time_report.start_timer("actor training")
            actor_loss = self.actor_optimizer.step(actor_closure)
            self.time_report.end_timer("actor training")

            # train critic
            # prepare dataset
            self.time_report.start_timer("prepare critic dataset")
            with torch.no_grad():
                self.compute_target_values()
                if self.avantage_objective:
                    dataset = CriticAdvDataset(self.batch_size, self.obs_buf, self.act_buf, self.target_values, self.advantages, drop_last = False)
                else:
                    dataset = CriticDataset(self.batch_size, self.obs_buf, self.target_values, drop_last = False)
            self.time_report.end_timer("prepare critic dataset")

            self.time_report.start_timer("critic training")
            self.value_loss = 0.
            for j in range(self.critic_iterations):
                total_critic_loss = 0.
                batch_cnt = 0
                for i in range(len(dataset)):
                    batch_sample = dataset[i]
                    self.critic_optimizer.zero_grad()
                    training_critic_loss = self.compute_critic_loss(batch_sample)
                    training_critic_loss.backward()
                    
                    # ugly fix for simulation nan problem
                    for params in self.critic.parameters():
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)

                    if self.truncate_grad:
                        clip_grad_norm_(self.critic.parameters(), self.grad_norm)

                    self.critic_optimizer.step()

                    total_critic_loss += training_critic_loss
                    batch_cnt += 1
                
                self.value_loss = (total_critic_loss / batch_cnt).detach().cpu().item()
                print('value iter {}/{}, loss = {:7.6f}'.format(j + 1, self.critic_iterations, self.value_loss), end='\r')

            self.time_report.end_timer("critic training")

            self.iter_count += 1
            
            time_end_epoch = time.time()

            # logging
            time_elapse = time.time() - self.start_time
            self.writer.add_scalar('lr/iter', lr, self.iter_count)
            self.writer.add_scalar('actor_loss/step', self.actor_loss, self.step_count)
            self.writer.add_scalar('actor_loss/iter', self.actor_loss, self.iter_count)
            #self.writer.add_scalar('dyn_loss/step', self.dyn_loss, self.step_count) ###
            #self.writer.add_scalar('dyn_loss/iter', self.dyn_loss, self.iter_count) ###
            self.writer.add_scalar('value_loss/step', self.value_loss, self.step_count)
            self.writer.add_scalar('value_loss/iter', self.value_loss, self.iter_count)
            #self.writer.add_scalar('discarded_ratio/step', self.discarded_ratio, self.step_count) ##
            if len(self.episode_loss_his) > 0:
                mean_episode_length = self.episode_length_meter.get_mean()
                mean_policy_loss = self.episode_loss_meter.get_mean()
                mean_policy_discounted_loss = self.episode_discounted_loss_meter.get_mean()

                if mean_policy_loss < self.best_policy_loss:
                    print_info("save best policy with loss {:.2f}".format(mean_policy_loss))
                    self.save()
                    self.best_policy_loss = mean_policy_loss
                
                self.writer.add_scalar('policy_loss/step', mean_policy_loss, self.step_count)
                self.writer.add_scalar('policy_loss/time', mean_policy_loss, time_elapse)
                self.writer.add_scalar('policy_loss/iter', mean_policy_loss, self.iter_count)
                self.writer.add_scalar('rewards/step', -mean_policy_loss, self.step_count)
                self.writer.add_scalar('rewards/time', -mean_policy_loss, time_elapse)
                self.writer.add_scalar('rewards/iter', -mean_policy_loss, self.iter_count)
                self.writer.add_scalar('policy_discounted_loss/step', mean_policy_discounted_loss, self.step_count)
                self.writer.add_scalar('policy_discounted_loss/iter', mean_policy_discounted_loss, self.iter_count)
                self.writer.add_scalar('best_policy_loss/step', self.best_policy_loss, self.step_count)
                self.writer.add_scalar('best_policy_loss/iter', self.best_policy_loss, self.iter_count)
                self.writer.add_scalar('episode_lengths/iter', mean_episode_length, self.iter_count)
                self.writer.add_scalar('episode_lengths/step', mean_episode_length, self.step_count)
                self.writer.add_scalar('episode_lengths/time', mean_episode_length, time_elapse)
            else:
                mean_policy_loss = np.inf
                mean_policy_discounted_loss = np.inf
                mean_episode_length = 0
            
            print('iter {}: ep loss {:.2f}, ep discounted loss {:.2f}, ep len {:.1f}, fps total {:.2f}, value loss {:.2f}, grad norm before clip {:.10f}, grad norm after clip {:.2f}, dyn loss {:.2f}'.format(\
                    self.iter_count, mean_policy_loss, mean_policy_discounted_loss, mean_episode_length, self.steps_num * self.num_envs / (time_end_epoch - time_start_epoch), self.value_loss, self.grad_norm_before_clip, self.grad_norm_after_clip, self.dyn_loss))

            self.writer.flush()
        
            if self.save_interval > 0 and (self.iter_count % self.save_interval == 0):
                self.save(self.name + "policy_iter{}_reward{:.3f}".format(self.iter_count, -mean_policy_loss))

            # update target critic
            with torch.no_grad():
                alpha = self.target_critic_alpha
                for param, param_targ in zip(self.critic.parameters(), self.target_critic.parameters()):
                    param_targ.data.mul_(alpha)
                    param_targ.data.add_((1. - alpha) * param.data)

        self.time_report.end_timer("algorithm")

        self.time_report.report()
        
        self.save('final_policy')

        # save reward/length history
        self.episode_loss_his = np.array(self.episode_loss_his)
        self.episode_discounted_loss_his = np.array(self.episode_discounted_loss_his)
        self.episode_length_his = np.array(self.episode_length_his)
        np.save(open(os.path.join(self.log_dir, 'episode_loss_his.npy'), 'wb'), self.episode_loss_his)
        np.save(open(os.path.join(self.log_dir, 'episode_discounted_loss_his.npy'), 'wb'), self.episode_discounted_loss_his)
        np.save(open(os.path.join(self.log_dir, 'episode_length_his.npy'), 'wb'), self.episode_length_his)

        # evaluate the final policy's performance
        self.run(self.num_envs)

        self.close()
    
    def play(self, cfg):
        self.load(cfg['params']['general']['checkpoint'])
        self.run(cfg['params']['config']['player']['games_num'])
        
    def save(self, filename = None):
        if filename is None:
            filename = 'best_policy'
        torch.save([self.actor, self.critic, self.target_critic, self.obs_rms, self.ret_rms], os.path.join(self.log_dir, "{}.pt".format(filename)))
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor = checkpoint[0].to(self.device)
        self.critic = checkpoint[1].to(self.device)
        self.target_critic = checkpoint[2].to(self.device)
        self.obs_rms = checkpoint[3].to(self.device)
        self.ret_rms = checkpoint[4].to(self.device) if checkpoint[4] is not None else checkpoint[4]
        
    def close(self):
        self.writer.close()
    
