import os
import sys

from utils_rl import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torch import nn
from torch.nn import functional as F

import time
from collections import namedtuple
from torch.distributions import Normal

Transition = namedtuple('Transition', ('state', 'action', 'next_state',
                                       'reward', 'mean', 'stddev', 'disc_rew', 'covariance'))

def collect_samples_mlp(pid, env, policy, num_ep, custom_reward, render, running_state, fixed_sigma):

    torch.randn(pid)
    log = dict()
    memory = Memory()  # every time we collect a batch he memory is re-initialized
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0
    with torch.no_grad():
        for ep in range(num_ep):

            episode = []
            reward_episode = 0

            state = env.reset()
            if running_state is not None:
                state = running_state(state)
            t_ep = time.time()
            for t in range(10000):
                state_var = tensor(state).unsqueeze(0).unsqueeze(0)
                pi = policy(state_var)
                mean = pi
                #stddev = pi.stddev
                sigma = fixed_sigma
                cov = torch.diag(sigma**2)

                action_distribution = Normal(mean, sigma)

                action = action_distribution.sample() # sample from normal distribution
                next_state, reward, done, _ = env.step(action.cpu())
                reward_episode += reward
                if running_state is not None:
                    next_state = running_state(next_state)
                if custom_reward is not None:
                    reward = custom_reward(state, action)
                    total_c_reward += reward
                    min_c_reward = min(min_c_reward, reward)
                    max_c_reward = max(max_c_reward, reward)

                episode.append(Transition(state, action.cpu().numpy(), next_state, reward, mean.cpu().numpy(),
                                          sigma.cpu().numpy(), None, cov))

                if render:
                    env.render()
                if done:
                    memory.push(episode)
                    break

                state = next_state
            # log stats
            num_steps += (t + 1)
            num_episodes += 1
            total_reward += reward_episode
            min_reward = min(min_reward, reward_episode)
            max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    return memory, log

class AgentMLP:

    def __init__(self, env, policy, num_episodes, device, custom_reward=None, attention=False,
                 render=False, running_state=None, fixed_sigma=None):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.running_state = running_state
        self.render = render
        self.attention = attention
        self.fixed_sigma = fixed_sigma
        self.num_episodes = num_episodes

    def collect_episodes(self):
        t_start = time.time()

        memory, log = collect_samples_mlp(0, self.env, self.policy, self.num_episodes, self.custom_reward,
                                      self.render, self.running_state, self.fixed_sigma)

        batch = memory.memory
        t_end = time.time()

        return memory, log


class MultiLayerPerceptron(nn.Module):
    """
    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    h_dim : int
        Dimension of hidden layer in encoder.
    """
    def __init__(self, x_dim, y_dim, h_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.h_dim = h_dim

        # Initialize networks
        layers = [nn.Linear(x_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim//2),
                  nn.ReLU(inplace=True)]

        self.x_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim//2, y_dim)

    def forward(self, x_target):
        """
        Parameters
        ----------
        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)

        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.

        """
        # Infer quantities from tensor dimensions
        batch_size, num_points, _ = x_target.size()
        # Flatten x and z to fit with linear layer
        x_flat = x_target.view(batch_size * num_points, self.x_dim)

        # Input is concatenation of the representation with every row of x
        hidden = self.x_to_hidden(x_flat)
        mu = self.hidden_to_mu(hidden)

        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)

        # Predict target points
        return mu

class MLPTrainer():
    """
    Parameters
    ----------
    device : torch.device

    model : MLPmodel

    optimizer : one of torch.optim optimizers

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, model, optimizer,print_freq=100):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.print_freq = print_freq
        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, epochs, early_stopping=None):
        """
        Trains MLP

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                x, y, num_points = data
                p_y_pred = self.model(x)
                loss = self._loss(p_y_pred, y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                self.steps += 1
            avg_loss = epoch_loss / len(data_loader)
            if epoch % self.print_freq == 0 or epoch == epochs-1:
                print("Epoch: {}, Avg_loss: {}".format(epoch, avg_loss))
            self.epoch_loss_history.append(avg_loss)

            if early_stopping is not None:
                if avg_loss < early_stopping:
                    break

    def _loss(self, y_pred, y_target):
        """
        Computes loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by MLP.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        """
        # square over batch and sum over number of targets and dimensions of y
        diff = (y_target - y_pred).transpose(0, 1)  # 1xNxD -> Nx1xD
        return diff.matmul(diff.transpose(1, 2)).sum()  # -> Nx1x1 -> 1

if __name__ == '__main__':
    import gym
    import torch
    from core.common import discounted_rewards, improvement_step_all, estimate_eta_3
    from utils_rl.memory_dataset import Memory, merge_padded_lists, get_close_context, ReplayMemoryDataset
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser(description='PyTorch TRPO example')
    parser.add_argument('--fixed-sigma', default=0.35, type=float, metavar='N',
                        help='sigma of the policy')
    parser.add_argument('--dtype', default=torch.float64,
                        help='default type')
    args = parser.parse_args()
    torch.set_default_tensor_type(torch.DoubleTensor)

    device = torch.device('cpu')
    env = gym.make('Hopper-v2')
    num_epi = 10

    max_episode_len = env._max_episode_steps
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    args.fixed_sigma = args.fixed_sigma * torch.ones(action_dim)

    model = MultiLayerPerceptron(state_dim, action_dim, 512).to(device).double()

    agent = AgentMLP(env, model, num_epi, device, fixed_sigma=args.fixed_sigma)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    model_trainer = MLPTrainer(device, model, optimizer, print_freq=50)
    replay_memory = ReplayMemoryDataset(20)
    tot_steps = [0]
    avg_rewards = [0]
    for i_iter in range(500):
        batch, log = agent.collect_episodes()  # batch of batch_size transitions from multiple

        disc_rew = discounted_rewards(batch.memory, 0.999)
        complete_dataset = BaseDataset(batch.memory, disc_rew, device, torch.float64,
                                          max_len=max_episode_len)
        print('average reward at', i_iter, log['avg_reward'].item())
        t0 = time.time()
        improved_context_list_mi = improvement_step_all(complete_dataset, disc_rew, 0.01, args)
        t1 = time.time()

        # create training set
        tn0 = time.time()
        replay_memory.add(complete_dataset)
        data_loader = DataLoader(replay_memory, batch_size=1, shuffle=True)
        model_trainer.train(data_loader, 30)
        tn1 = time.time()
        tot_steps.append(tot_steps[-1] + log['num_steps'])

        avg_rewards.append(log['avg_reward'].item())
        try:
            os.mkdir('/home/francesco/PycharmProjects/MasterThesis/mujoco learning results/mlp')
        except FileExistsError:
            pass
        fig = plt.figure()
        plt.plot(tot_steps[1:], avg_rewards[1:])
        fig.savefig('/home/francesco/PycharmProjects/MasterThesis/mujoco learning results/mlp/')
        plt.show()