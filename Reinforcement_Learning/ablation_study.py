import argparse
import gym
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import randint
from utils_rl import *
import gpytorch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.common import discounted_rewards
from torch import nn
from torch.nn import functional as F
from weights_init import InitFunc
torch.set_default_tensor_type(torch.DoubleTensor)
from neural_process import NeuralProcess
from training_module_RL import NeuralProcessTrainerRL
from multihead_attention_np import AttentiveNeuralProcess
from utils_rl.memory_dataset import Memory, merge_context, get_close_context
import time
from collections import namedtuple
from torch.distributions import Normal

Transition = namedtuple('Transition', ('state', 'action', 'next_state',
                                       'reward', 'mean', 'stddev', 'disc_rew'))

def collect_samples(pid, env, policy, num_ep, custom_reward, render, running_state, fixed_sigma):
    # (2)
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
                mean = pi.mean
                stddev = pi.stddev

                if fixed_sigma is not None:
                    sigma = fixed_sigma
                else:
                    sigma = stddev

                action_distribution = Normal(mean, sigma)

                action = action_distribution.sample().squeeze(0).squeeze(0)  # sample from normal distribution
                next_state, reward, done, _ = env.step(action)
                reward_episode += reward
                if running_state is not None:  # running list of normalized states allowing to access precise mean and std
                    next_state = running_state(next_state)
                if custom_reward is not None:  # by default is None, unless given when init Agent
                    reward = custom_reward(state, action)
                    total_c_reward += reward
                    min_c_reward = min(min_c_reward, reward)
                    max_c_reward = max(max_c_reward, reward)

                episode.append(Transition(state, action.numpy(), next_state, reward, mean.numpy(), stddev.numpy(), None))

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

def compute_stats(batch):
    s = 0
    ma = -100
    mi = 100
    l = 0
    for ep in batch:
        l += len(ep)
        for tr in ep:
            action = tr.action
            s += action
            ma = max(ma, action)
            mi = min(mi, action)
    avg = s / l
    return avg, ma, mi


class Agent:

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
        to_device(torch.device('cpu'), self.policy)

        memory, log = collect_samples(0, self.env, self.policy, self.num_episodes, self.custom_reward,
                                      self.render, self.running_state, self.fixed_sigma)

        batch = memory.memory
        to_device(self.device, self.policy)
        t_end = time.time()
        mean_a, max_a, min_a = compute_stats(batch)
        log['sample_time'] = t_end - t_start
        log['action_mean'] = mean_a
        log['action_min'] = min_a
        log['action_max'] = max_a
        return memory, log


class Decoder(nn.Module):
    """
    Maps target input x_target and samples z (encoding information about the
    context points) to predictions y_target.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    h_dim : int
        Dimension of hidden layer.

    y_dim : int
        Dimension of y values.
    """
    def __init__(self, x_dim, h_dim, y_dim, fixed_sigma):
        super(Decoder, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.fixed_sigma = fixed_sigma

        layers = [nn.Linear(x_dim, h_dim),
                  nn.ReLU(inplace=True),
                  nn.Linear(h_dim, h_dim),
                  nn.ReLU(inplace=True)]

        self.x_to_hidden = nn.Sequential(*layers)
        self.hidden_to_mu = nn.Linear(h_dim, y_dim)
        self.hidden_to_sigma = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        """
        x : torch.Tensor
            Shape (batch_size, num_points, x_dim)

        z : torch.Tensor
            Shape (batch_size, z_dim)

        Returns
        -------
        Returns mu and sigma for output distribution. Both have shape
        (batch_size, num_points, y_dim).
        """
        batch_size, num_points, _ = x.size()
        # Flatten x and z to fit with linear layer
        x_flat = x.view(batch_size * num_points, self.x_dim)
        # Input is concatenation of the representation with every row of x
        hidden = self.x_to_hidden(x_flat)
        mu = self.hidden_to_mu(hidden)
        pre_sigma = self.hidden_to_sigma(hidden)
        # Reshape output into expected shape
        mu = mu.view(batch_size, num_points, self.y_dim)
        pre_sigma = pre_sigma.view(batch_size, num_points, self.y_dim)
        # Define sigma following convention in "Empirical Evaluation of Neural
        # Process Objectives" and "Attentive Neural Processes"
        if self.fixed_sigma is None:
            sigma = 0.1 + 0.9 * F.softplus(pre_sigma)
        else:
            sigma = torch.Tensor(mu.shape)
            sigma.fill_(self.fixed_sigma)

        return mu, sigma

class NeuralProcessAblated(nn.Module):
    """
    Implements Neural Process for functions of arbitrary dimensions.

    Parameters
    ----------
    x_dim : int
        Dimension of x values.

    y_dim : int
        Dimension of y values.

    r_dim : int
        Dimension of output representation r.


    h_dim : int
        Dimension of hidden layer in encoder and decoder.
    """
    def __init__(self, x_dim, y_dim, r_dim, h_dim, fixed_sigma=None):
        super(NeuralProcessAblated, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.r_dim = r_dim
        self.h_dim = h_dim
        self.fixed_sigma = fixed_sigma

        # Initialize networks
        self.x_to_y = Decoder(x_dim, h_dim, y_dim, fixed_sigma)

    def aggregate(self, r_i):
        """
        Aggregates representations for every (x_i, y_i) pair into a single
        representation.

        Parameters
        ----------
        r_i : torch.Tensor
            Shape (batch_size, num_points, r_dim)
        """
        return torch.mean(r_i, dim=1)

    def forward(self, x_target, y_target=None):
        """
        Given context pairs (x_context, y_context) and target points x_target,
        returns a distribution over target points y_target.

        Parameters
        ----------
        x_target : torch.Tensor
            Shape (batch_size, num_target, x_dim)

        y_target : torch.Tensor or None
            Shape (batch_size, num_target, y_dim). Only used during training.

        Note
        ----
        We follow the convention given in "Empirical Evaluation of Neural
        Process Objectives" where context is a subset of target points. This was
        shown to work best empirically.
        """
        # Infer quantities from tensor dimensions
        _, num_target, _ = x_target.size()

        if self.training:
            _, _, y_dim = y_target.size()

            y_pred_mu, y_pred_sigma = self.x_to_y(x_target)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)

            return p_y_pred
        else:
            # Predict target points
            y_pred_mu, y_pred_sigma = self.x_to_y(x_target)
            p_y_pred = Normal(y_pred_mu, y_pred_sigma)
            return p_y_pred

class TrainerAblated():
    """
    Class to handle training of Neural Processes and Attentive Neural Process
    for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or neural_process.AttentiveNeuralProcess
                     or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : not used

    num_extra_target_range : not used

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer,print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.print_freq = print_freq
        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, epochs, early_stopping=None):
        """
        Trains Neural Process.

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
                p_y_pred = self.neural_process(x, y)
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

    def _loss(self, p_y_pred, y_target):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        return -log_likelihood

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="MountainCarContinuous-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')

parser.add_argument('--use-running-state', default=False,
                    help='store running mean and variance instead of states and actions')
parser.add_argument('--max-kl', type=float, default=0.2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--num-ensembles', type=int, default=10, metavar='N',
                    help='episode to collect per iteration')
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')

parser.add_argument('--use-mean', default=True, metavar='N',
                    help='train & condit on improved means/actions'),
parser.add_argument('--fixed-sigma', default=1., metavar='N', type=float,
                    help='sigma of the policy')
parser.add_argument('--epochs-per-iter', type=int, default=20, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--replay-memory-size', type=int, default=10, metavar='G',
                    help='size of training set in episodes ')
parser.add_argument('--z-dim', type=int, default=128, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--r-dim', type=int, default=128, metavar='N',
                    help='dimension of representation space in np')
parser.add_argument('--h-dim', type=int, default=128, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--np-batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--early-stopping', type=int, default=-10, metavar='N',
                    help='stop training training when avg_loss reaches it')

parser.add_argument('--v-epochs-per-iter', type=int, default=20, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--v-replay-memory-size', type=int, default=5, metavar='G',
                    help='size of training set in episodes')
parser.add_argument('--v-z-dim', type=int, default=128, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--v-r-dim', type=int, default=128, metavar='N',
                    help='dimension of representation space in np')
parser.add_argument('--v-h-dim', type=int, default=128, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--v-np-batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--v-early-stopping', type=int, default=0, metavar='N',
                    help='stop training training when avg_loss reaches it')

parser.add_argument('--directory-path', default='/home/francesco/PycharmProjects/MasterThesis/NP learning results/',
                    help='path to plots folder')
parser.add_argument('--device-np', default=torch.device('cpu'),
                    help='device')
parser.add_argument('--dtype', default=torch.float64,
                    help='default type')
parser.add_argument('--seed', type=int, default=7, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')

parser.add_argument('--use-attentive-np', default=True, metavar='N',
                     help='use attention in policy and value NPs')
parser.add_argument('--v-use-attentive-np', default=True, metavar='N',
                     help='use attention in policy and value NPs')
parser.add_argument('--episode-specific-value', default=False, metavar='N',
                    help='condition the value np on all episodes')
parser.add_argument("--plot-every", type=int, default=5,
                    help='plot every n iter')
parser.add_argument("--num-testing-points", type=int, default=1,
                    help='how many point to use as only testing during NP training')
args = parser.parse_args()
initial_training = True
init_func = InitFunc.init_zero

max_episode_len = 999
num_context_points = max_episode_len - args.num_testing_points

np_spec = '_{}z_{}rm_{}vrm_{}e_num_context:{}_earlystop{}|{}'.format(args.z_dim, args.replay_memory_size, args.v_replay_memory_size,
                                                       args.epochs_per_iter, num_context_points, args.early_stopping, args.v_early_stopping)
run_id = '/ABLATION_mean:{}_A_p:{}_A_v:{}_fixSTD:{}_epV:{}_{}ep_{}kl_{}gamma_'.format(args.use_mean,
                                                args.use_attentive_np,  args.v_use_attentive_np, args.fixed_sigma, args.episode_specific_value,
                                                args.num_ensembles, args.max_kl, args.gamma) + np_spec
args.directory_path += run_id

torch.set_default_dtype(args.dtype)

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
if args.use_running_state:
    running_state = ZFilter((state_dim,), clip=5)  # running list of states that allows to access precise mean and std
else:
    running_state = None

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

'''create neural process'''
if args.use_attentive_np:
    print('Use NP anyways')

policy_np = NeuralProcessAblated(state_dim, action_dim, args.r_dim, args.h_dim).to(args.device_np)

optimizer = torch.optim.Adam(policy_np.parameters(), lr=3e-4)
np_trainer = TrainerAblated(args.device_np, policy_np, optimizer, print_freq=50)

if args.v_use_attentive_np:
    value_np = AttentiveNeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_h_dim,
                                      args.z_dim, use_self_att=False).to(args.device_np)
else:
    value_np = NeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_h_dim).to(args.device_np)
value_optimizer = torch.optim.Adam(value_np.parameters(), lr=3e-4)
value_np_trainer = NeuralProcessTrainerRL(args.device_np, value_np, value_optimizer,
                                          num_context_range=(num_context_points, num_context_points),
                                          num_extra_target_range=(args.num_testing_points, args.num_testing_points),
                                          print_freq=50)
"""create replay memory"""
# force rm to contain only last iter episodes
replay_memory = ReplayMemoryDataset(args.num_ensembles, use_mean=args.use_mean)
value_replay_memory = ValueReplay(args.v_replay_memory_size)

"""create agent"""
agent = Agent(env, policy_np, args.num_ensembles, args.device_np, render=args.render,
              attention=args.use_attentive_np, fixed_sigma=args.fixed_sigma)

def estimate_eta_2(actions, means, stddevs, disc_rews):
    """Compute learning step for an episode"""
    d = actions.shape[-1]
    if d > 1:
        raise NotImplementedError('compute eta not implemented for action space of dim>1')
    else:
        stddev = args.fixed_sigma
        iter_sum = 0
        eps = tensor(args.max_kl).to(args.dtype)
        T = tensor(actions.shape[0]).to(args.dtype)
        for action, mean, stddev_np, disc_reward in zip(actions, means, stddevs, disc_rews):
            iter_sum += ((disc_reward ** 2) * (action - mean) ** 2) / (2*(stddev ** 4))
        denominator = iter_sum.to(args.dtype)
        return torch.sqrt((T*eps)/denominator)

def estimate_eta_3(actions, means, advantages):
    """Compute learning step from all the samples of previous iteration"""
    d = actions.shape[-1]
    if d > 1:
        raise NotImplementedError('compute eta not implemented for action space of dim>1')
    else:
        stddev = args.fixed_sigma
        iter_sum = 0
        eps = tensor(args.max_kl).to(args.dtype)
        T = tensor(actions.shape[0]).to(args.dtype)
        for action, mean, disc_reward in zip(actions, means, advantages):
            iter_sum += ((disc_reward ** 2) * (action - mean) ** 2) / (2 * (stddev ** 4))
        denominator = iter_sum.to(args.dtype)
        return torch.sqrt((T * eps) / denominator)


def improvement_step_all(complete_dataset, estimated_adv):
    """Perform improvement step using same eta for all episodes"""
    all_improved_context = []
    with torch.no_grad():
        all_states, all_means, all_actions = merge_padded_lists([episode['states'] for episode in complete_dataset],
                                                                [episode['means'] for episode in complete_dataset],
                                                                [episode['actions'] for episode in complete_dataset],
                                                                 max_lens=[episode['real_len'] for episode in complete_dataset])
        all_advantages = [adv for ep in estimated_adv for adv in ep]
        eta = estimate_eta_3(all_actions, all_means, all_advantages)
        for episode, episode_adv in zip(complete_dataset, estimated_adv):
            real_len = episode['real_len']
            states = episode['states'][:real_len]
            actions = episode['actions'][:real_len]
            means = episode['means'][:real_len]
            new_padded_actions = torch.zeros_like(episode['actions'])
            new_padded_means = torch.zeros_like(episode['means'])
            i = 0
            for state, action, mean, advantage in zip(states, actions, means, episode_adv):
                new_mean = mean + eta * advantage * ((action - mean) / args.fixed_sigma)
                distr = Normal(new_mean, args.fixed_sigma)
                new_action = distr.sample()
                new_padded_actions[i, :] = new_action
                new_padded_means[i, :] = new_mean
                i += 1
            episode['new_means'] = new_padded_means
            episode['new_actions'] = new_padded_actions
            if args.use_mean:
                all_improved_context.append([episode['states'].unsqueeze(0), new_padded_means.unsqueeze(0), real_len])
            else:
                all_improved_context.append([episode['states'].unsqueeze(0), new_padded_actions.unsqueeze(0), real_len])

    return all_improved_context

def improvement_step(complete_dataset, estimated_disc_rew):
    """Perform improvement step"""
    all_improved_context = []
    with torch.no_grad():
        for episode, disc_rewards in zip(complete_dataset, estimated_disc_rew):
            real_len = episode['real_len']
            states = episode['states'][:real_len]
            actions = episode['actions'][:real_len]
            means = episode['means'][:real_len]
            stddevs = episode['stddevs'][:real_len]
            eta = estimate_eta_2(actions, means, stddevs, disc_rewards)
            new_padded_actions = torch.zeros_like(episode['actions'])
            new_padded_means = torch.zeros_like(episode['means'])
            i = 0
            for state, action, mean, stddev, disc_reward in zip(states, actions, means, stddevs, disc_rewards):
                new_mean = mean + eta * disc_reward * ((action - mean) / args.fixed_sigma)
                distr = Normal(new_mean, args.fixed_sigma)
                new_action = distr.sample()
                new_padded_actions[i, :] = new_action
                new_padded_means[i, :] = new_mean
                i += 1
            episode['new_means'] = new_padded_means
            episode['new_actions'] = new_padded_actions
            if args.use_mean:
                all_improved_context.append([episode['states'].unsqueeze(0), new_padded_means.unsqueeze(0), real_len])
            else:
                all_improved_context.append([episode['states'].unsqueeze(0), new_padded_actions.unsqueeze(0), real_len])

    return all_improved_context


def train_np(datasets, epochs=args.epochs_per_iter):
    print('Policy training')
    policy_np.training = True
    data_loader = DataLoader(datasets, batch_size=args.np_batch_size, shuffle=True)
    np_trainer.train(data_loader, epochs, early_stopping=args.early_stopping)
    policy_np.training = False


def train_value_np(value_replay_memory):
    print('Value training')
    value_np.training = True
    value_data_loader = DataLoader(value_replay_memory, batch_size=args.v_np_batch_size, shuffle=True)
    value_np_trainer.train(value_data_loader, args.v_epochs_per_iter, early_stopping=args.v_early_stopping)
    value_np.training = False


def estimate_disc_rew(all_episodes, i_iter, episode_specific_value=False):
    estimated_disc_rew = []
    value_stddevs = []
    if episode_specific_value:
        for episode in all_episodes.data:
            real_len = episode['real_len']
            x = episode['states'][:real_len].unsqueeze(0)
            context_y = episode['discounted_rewards'][:real_len].unsqueeze(0)
            with torch.no_grad():
                values_distr = value_np(x, context_y, x)
                values = values_distr.mean
                r_est = context_y - values
                estimated_disc_rew.append(r_est.view(-1).numpy())
                value_stddevs.append(values_distr.stddev.view(-1).numpy())
        all_states = x
        all_values = [values]
        all_episodes = [all_episodes[-1]]
        all_rewards = context_y
    else:
        real_len = all_episodes[0]['real_len']
        all_states = all_episodes.data[0]['states'][:real_len]
        all_rewards = all_episodes.data[0]['discounted_rewards'][:real_len]
        for episode in all_episodes.data[1:]:
            real_len = episode['real_len']
            states = episode['states'][:real_len]
            rewards = episode['discounted_rewards'][:real_len]
            all_states = torch.cat((all_states, states), dim=0)
            all_rewards = torch.cat((all_rewards, rewards), dim=0)
        all_states = all_states.unsqueeze(0)
        all_rewards = all_rewards.unsqueeze(0)
        all_values = []
        for episode in all_episodes.data:
            real_len = episode['real_len']
            x = episode['states'][:real_len].unsqueeze(0)
            context_y = episode['discounted_rewards'][:real_len].unsqueeze(0)
            with torch.no_grad():
                values_distr = value_np(all_states, all_rewards, x)
                values = values_distr.mean
                r_est = context_y - values
                estimated_disc_rew.append(r_est.view(-1).numpy())
                value_stddevs.append(values_distr.stddev.view(-1).numpy())
            all_values.append(values)
    if i_iter % args.plot_every == 0:
        plot_NP_value(value_np, all_states, all_values, all_episodes, all_rewards, value_replay_memory, env, args, i_iter)
    return estimated_disc_rew, value_stddevs


def sample_initial_context_normal(num_episodes):
    initial_episodes = []
    policy_np.apply(init_func)
    sigma = 0.5
    if args.fixed_sigma is not None:
        sigma = args.fixed_sigma
    for e in range(num_episodes):
        states = torch.zeros([1, max_episode_len, state_dim])

        for i in range(max_episode_len):
            states[:, i, :] = torch.from_numpy(env.observation_space.sample())

        if args.use_attentive_np:
            dims = [1, max_episode_len, action_dim]
            distr_init = Normal(zeros(dims), sigma*ones(dims))
            actions_init = distr_init.sample()
        else:
            z_sample = torch.randn((1, args.z_dim)).unsqueeze(1).repeat(1, max_episode_len, 1)
            means_init, stds_init = policy_np.xz_to_y(states, z_sample)
            actions_init = Normal(means_init, stds_init).sample()
        initial_episodes.append([states, actions_init, max_episode_len])
    return initial_episodes

def train_on_initial(initial_context_list):
    #print('training on initial context')
    train_list = []
    for episode in initial_context_list:
        train_list.append([episode[0].squeeze(0), episode[1].squeeze(0), episode[2]])

    train_np(train_list, 2*args.epochs_per_iter)


def create_directories(directory_path):

    os.mkdir(directory_path)
    os.mkdir(directory_path + '/policy/')
    os.mkdir(directory_path + '/value/')
    os.mkdir(directory_path + '/policy/'+'/NP estimate/')
    os.mkdir(directory_path + '/policy/' + '/Mean improvement/')
    os.mkdir(directory_path + '/policy/' + '/Training/')
    os.mkdir(directory_path + '/policy/' + '/All policies samples/')

avg_rewards = []


def main_loop():
    colors = []
    num_episodes = args.num_ensembles
    for i in range(num_episodes):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    improved_context_list = sample_initial_context_normal(args.num_ensembles)
    plot_initial_context(improved_context_list, colors, env, args, '00')
    if initial_training:
        train_on_initial(improved_context_list)
    for i_iter in range(args.max_iter_num):
        print('sampling episodes')
        # (1)
        # generate multiple trajectories that reach the minimum batch_size
        policy_np.training = False
        batch, log = agent.collect_episodes()  # batch of batch_size transitions from multiple
        #print(log['num_steps'], log['num_episodes'])                # episodes (separated by mask=0). Stored in Memory

        disc_rew = discounted_rewards(batch.memory, args.gamma)
        complete_dataset = BaseDataset(batch.memory, disc_rew, args.device_np, args.dtype,  max_len=max_episode_len)
        if not args.episode_specific_value:
            iter_dataset = {}
            iter_states, iter_q = merge_padded_lists([episode['states'] for episode in complete_dataset],
                                                     [episode['discounted_rewards'] for episode in complete_dataset],
                                                     max_lens=[episode['real_len'] for episode in complete_dataset])
            iter_dataset['states'] = iter_states
            iter_dataset['discounted_rewards'] = iter_q
            iter_dataset['real_len'] = iter_states.shape[-2]
            value_replay_memory.add([iter_dataset])
        else:
            value_replay_memory.add(complete_dataset)
        tv0 = time.time()
        train_value_np(value_replay_memory)
        tv1 = time.time()
        estimated_disc_rew, values_stdevs = estimate_disc_rew(complete_dataset, i_iter, episode_specific_value=args.episode_specific_value)

        t0 = time.time()
        improved_context_list = improvement_step_all(complete_dataset, estimated_disc_rew)
        t1 = time.time()
        #plot_initial_context(improved_context_list, colors, env, args, i_iter)
        # plot improved context and actions' discounted rewards
        if i_iter % args.plot_every == 0:
            plot_improvements(complete_dataset, estimated_disc_rew, env, i_iter, args, colors)

        # create training set
        tn0 = time.time()
        replay_memory.add(complete_dataset)
        train_np(replay_memory)
        tn1 = time.time()

        #plot_training_set(i_iter, replay_memory, env, args)
        if i_iter % args.plot_every == 0:
            plot_NP_policy(policy_np, args.num_ensembles, replay_memory, i_iter, log['avg_reward'], env, args, colors)

        avg_rewards.append(log['avg_reward'])
        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f} \tT_update {:.4f} \tR_min {:.2f} \tR_max {:.2f} \tR_avg {:.2f}'.format(
                  i_iter, log['sample_time'], t1 - t0, log['min_reward'], log['max_reward'], log['avg_reward']))
            print('Training:  \tT_policy {:.2f}  \tT_value {:.2f}'.format(tn1-tn0, tv1-tv0))
        if log['avg_reward'] > 95:
            print('converged')
            plot_rewards_history(avg_rewards, args)
        args.fixed_sigma = args.fixed_sigma * args.gamma
    plot_rewards_history(avg_rewards, args)

    """clean up gpu memory"""
    torch.cuda.empty_cache()


def set_labels(ax, np_id='policy'):
    ax.set_xlabel('Position', fontsize=14)
    ax.set_ylabel('Velocity', fontsize=14)
    if np_id == 'value':
        ax.set_zlabel('Reward', fontsize=14)
    else:
        ax.set_zlabel('Acceleration', fontsize=14)

def set_limits(ax, env, args, np_id='policy'):
    bounds_high = env.observation_space.high
    bounds_low = env.observation_space.low
    if np_id == 'value':
        ax.set_xlim(bounds_low[0], bounds_high[0])
        ax.set_ylim(bounds_low[1], bounds_high[1])
        return
    if args.use_running_state:
        ax.set_zlim(env.action_space.low, env.action_space.high)
        return

    ax.set_xlim(bounds_low[0], bounds_high[0])
    ax.set_ylim(bounds_low[1], bounds_high[1])
    ax.set_zlim(env.action_space.low, env.action_space.high)

def create_plot_grid(env, args, size=20):
    bounds_high = env.observation_space.high
    bounds_low = env.observation_space.low
    if not args.use_running_state:
        x1 = np.linspace(bounds_low[0], bounds_high[0], size)
        x2 = np.linspace(bounds_low[1], bounds_high[1], size)
    else:
        x1 = np.linspace(-2, 2, size)
        x2 = np.linspace(-2, 2, size)
    X1, X2 = np.meshgrid(x1, x2)

    grid = torch.zeros(size, 2)
    for i in range(2):
        grid_diff = float(bounds_high[i] - bounds_low[i]) / (size - 2)
        grid[:, i] = torch.linspace(bounds_low[i] - grid_diff, bounds_high[i] + grid_diff, size)

    x = gpytorch.utils.grid.create_data_from_grid(grid)
    x = x.unsqueeze(0).to(args.dtype).to(args.device_np)
    return x, X1, X2, x1, x2

def plot_NP_policy(policy_np, num_ep, rm, iter_pred, avg_rew, env, args, colors):
    num_test_context = 999
    policy_np.training = False
    x, X1, X2, x1, x2 = create_plot_grid(env, args)

    # set up axes
    name = 'NP policy for iteration {}, avg rew {}'.format(iter_pred, int(avg_rew))
    fig = plt.figure(figsize=(16, 14))  # figsize=plt.figaspect(1.5)
    fig.suptitle(name, fontsize=20)
    fig.tight_layout()

    ax_context = fig.add_subplot(222, projection='3d')
    set_labels(ax_context)
    set_limits(ax_context, env, args)
    ax_context.set_title('Context points improved', pad=30, fontsize=16)

    ax_samples = fig.add_subplot(223, projection='3d')
    ax_samples.set_title('Samples from policy, fixed sigma {:.2f}'.format(args.fixed_sigma), pad=30, fontsize=16)
    set_limits(ax_samples, env, args)
    set_labels(ax_samples)

    stddev_low_list = []
    stddev_high_list = []
    z_distr_list = []
    for e in range(num_ep):
        with torch.no_grad():
            z_distr = policy_np(x)  # B x num_points x z_dim  (B=1)
            z_distr_list.append(z_distr)
            z_mean = z_distr.mean.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
            z_stddev = z_distr.stddev.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
            stddev_low_list.append(z_mean - z_stddev)
            stddev_high_list.append(z_mean + z_stddev)

    ax_mean = fig.add_subplot(221, projection='3d')
    for stddev_low, stddev_high in zip(stddev_low_list, stddev_high_list):
        i = 0
        for y_slice in x2:
            ax_mean.add_collection3d(
                plt.fill_between(x1, stddev_low[i, :].cpu(), stddev_high[i, :].cpu(), color='lightseagreen',
                                 alpha=0.01),
                zs=y_slice, zdir='y')
            i += 1
    for e, z_distr in enumerate(z_distr_list):
        z_mean = z_distr.mean.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
        ax_mean.plot_surface(X1, X2, z_mean.cpu().numpy(), cmap='viridis', vmin=-1., vmax=1.)

        a_distr = Normal(z_distr.mean, args.fixed_sigma*torch.ones_like(z_distr.mean))
        z_sample = a_distr.sample().detach()[0].reshape(X1.shape)
        ax_samples.plot_surface(X1, X2, z_sample.cpu().numpy(), color=colors[e], alpha=0.2)

    ax_rm = fig.add_subplot(224, projection='3d')
    set_limits(ax_rm, env, args)
    set_labels(ax_rm)
    ax_rm.set_title('Training set (RM)', pad=30, fontsize=16)
    for traj in rm:
        r_len = traj[-1]
        ax_rm.scatter(traj[0][:r_len, 0].detach(), traj[0][:r_len, 1].detach(), traj[1][:r_len, 0].detach(),
                      c=traj[1][:r_len, 0].detach(), alpha=0.1, cmap='viridis')
    set_labels(ax_mean)
    set_limits(ax_mean, env, args)
    ax_mean.set_title('Mean of the NP policies', pad=30, fontsize=16)
    # plt.show()
    fig.savefig(args.directory_path + '/policy/'+'/NP estimate/'+name, dpi=250)
    plt.close(fig)


def plot_training_set(i_iter, replay_memory, env, args):
    name = 'Training trajectories ' + str(i_iter)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_title(name)
    set_limits(ax, env, args)
    set_labels(ax)
    for trajectory in replay_memory:
        real_len = trajectory[2]
        z = trajectory[1][:real_len,0].detach().cpu().numpy()
        xs_context = trajectory[0][:real_len,0].detach().cpu().numpy()
        ys_context = trajectory[0][:real_len,1].detach().cpu().numpy()
        ax.scatter(xs_context, ys_context, z, c=z, cmap='viridis', alpha=0.1, vmin=-1., vmax=1.)
    fig.savefig(args.directory_path + '/policy/'+'/Training/'+name)
    plt.close(fig)

def plot_improvements(all_dataset, est_rewards, env, i_iter, args, colors):

    name = 'Improvement iter ' + str(i_iter)
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(name, fontsize=20)
    ax = fig.add_subplot(121, projection='3d')
    name_c = 'Context improvement iter ' + str(i_iter)
    ax.set_title(name_c)
    set_limits(ax, env, args)
    set_labels(ax)
    ax_rew = fig.add_subplot(122, projection='3d')
    set_labels(ax_rew)
    set_limits(ax_rew, env, args)
    for e, episode in enumerate(all_dataset):
        real_len = episode['real_len']
        states = episode['states'][:real_len]
        disc_rew = episode['discounted_rewards'][:real_len]
        actions = episode['actions'][:real_len]
        means = episode['means'][:real_len]
        new_means = episode['new_means'][:real_len]
        est_rew = est_rewards[e]
        if e == 0:
            ax.scatter(states[:, 0].numpy(), states[:, 1].numpy(), means[:, 0].numpy(), c='k', s=4, label='sampled', alpha=0.3)
            ax.scatter(states[:, 0].numpy(), states[:, 1].numpy(), new_means[:, 0].numpy(), c=new_means[:, 0].numpy(), marker='+', label='improved', alpha=0.3)
            leg = ax.legend(loc="upper right")
        else:
            ax.scatter(states[:, 0].numpy(), states[:, 1].numpy(), means[:, 0].numpy(), c='k', s=4, alpha=0.3)
            ax.scatter(states[:, 0].numpy(), states[:, 1].numpy(), new_means[:, 0].numpy(), c=new_means[:, 0].numpy(), marker='+', alpha=0.3)

        a = ax_rew.scatter(states[:, 0].numpy(), states[:, 1].numpy(), actions[:, 0].numpy(), c=est_rew[:], cmap='viridis', alpha=0.5)

    cb = fig.colorbar(a)
    cb.set_label('Discounted rewards')
    ax_rew.set_title('Discounted rewards')
    fig.savefig(args.directory_path+'/policy/Mean improvement/'+name, dpi=250)
    plt.close(fig)


def plot_rewards_history(avg_rewards, args):
    fig_rew, ax_rew = plt.subplots(1, 1)
    ax_rew.plot(np.arange(len(avg_rewards)), avg_rewards)
    ax_rew.set_xlabel('iterations')
    ax_rew.set_ylabel('average reward')
    ax_rew.set_title('Average Reward History')
    plt.grid()
    fig_rew.savefig(args.directory_path + '/average reward')
    plt.close(fig_rew)

def plot_NP_value(value_np, all_states, all_values, all_episodes, all_rews, rm, env, args, i_iter):
    name = 'Value estimate ' + str(i_iter)
    fig = plt.figure(figsize=(16, 14))  # figsize=plt.figaspect(1.5)
    fig.suptitle(name, fontsize=20)
    fig.tight_layout()
    value_np.training = False
    x, X1, X2, x1, x2 = create_plot_grid(env, args)
    # Plot a realization
    with torch.no_grad():
        V_distr = value_np(all_states, all_rews, x)  # B x num_points x z_dim  (B=1)
        V_mean = V_distr.mean.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
        V_stddev = V_distr.stddev.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
    stddev_low = V_mean - V_stddev
    stddev_high = V_mean + V_stddev
    vmin = stddev_low.min()
    vmax = stddev_high.max()
    ax_mean = fig.add_subplot(221, projection='3d')
    i = 0
    for y_slice in x2:
        ax_mean.add_collection3d(
            plt.fill_between(x1, stddev_low[i, :].cpu(), stddev_high[i, :].cpu(), color='lightseagreen', alpha=0.05),
            zs=y_slice, zdir='y')
        i += 1
    ax_mean.plot_surface(X1, X2, V_mean.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
    set_labels(ax_mean, np_id='value')
    set_limits(ax_mean, env, args, np_id='value')
    ax_mean.set_zlim(vmin, vmax)

    ax_mean.set_title('Mean and std. dev. of the value NP', pad=25, fontsize=16)

    ax_context = fig.add_subplot(222, projection='3d')
    set_limits(ax_context, env, args, np_id='value')
    set_labels(ax_context, np_id='value')
    ax_context.set_title('Context and  prediction', pad=25, fontsize=16)

    ax_diff = fig.add_subplot(223, projection='3d')
    set_limits(ax_diff, env, args, np_id='value')
    set_labels(ax_diff, np_id='value')
    ax_diff.set_title('Q - V advantage estimate', pad=25, fontsize=16)
    first = True
    for episode, values in zip(all_episodes, all_values):
        real_len = episode['real_len']
        states = episode['states'][:real_len].unsqueeze(0)
        disc_rew = episode['discounted_rewards'][:real_len].unsqueeze(0)
        r_est = disc_rew - values
        if first:
            ax_context.scatter(states[0, :, 0].detach(), states[0, :, 1].detach(), disc_rew[0, :, 0].detach(), c='r',
                               label='Discounted rewards (~Q)', alpha=0.1)
            ax_context.scatter(states[0, :, 0].detach(), states[0, :, 1].detach(), values[0, :, 0].detach(), c='b',
                               label='Estimated values (~V)', alpha=0.1)
        else:
            ax_context.scatter(states[0, :, 0].detach(), states[0, :, 1].detach(), disc_rew[0, :, 0].detach(), c='r',alpha=0.1)
            ax_context.scatter(states[0, :, 0].detach(), states[0, :, 1].detach(), values[0, :, 0].detach(), c='b',alpha=0.1)

        ax_diff.scatter(states[0, :, 0].detach(), states[0, :, 1].detach(), r_est[0, :, 0].detach(), c='g', alpha=0.08)
        first = False
    leg = ax_context.legend(loc="upper right")
    ax_rm = fig.add_subplot(224, projection='3d')
    set_limits(ax_rm, env, args, np_id='value')
    set_labels(ax_rm, np_id='value')
    ax_rm.set_title('Training set (RM)', pad=25, fontsize=16)
    for traj in rm:
        r_len = traj[-1]
        ax_rm.scatter(traj[0][:r_len, 0].detach(), traj[0][:r_len, 1].detach(), traj[1][:r_len, 0].detach(),
                           c=traj[1][:r_len, 0].detach(), alpha=0.1, cmap='viridis')
    fig.savefig(args.directory_path+'/value/'+name)
    plt.close(fig)


def plot_initial_context(context_points, colors, env, args, i_iter):
    name = 'Contexts of all episodes at iter ' + str(i_iter)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.set_title(name)
    #set_limits(ax, env, args)
    set_labels(ax)
    for e, episode in enumerate(context_points):
        real_len = episode[2]
        z = episode[1][:, :real_len, 0].detach().cpu().numpy()
        xs_context = episode[0][:, :real_len, 0].detach().cpu().numpy()
        ys_context = episode[0][:, :real_len, 1].detach().cpu().numpy()
        ax.scatter(xs_context, ys_context, z, c=colors[e], alpha=0.5)
    fig.savefig(args.directory_path + '/policy/'+ '/All policies samples/' + name)
    plt.close(fig)



create_directories(args.directory_path)
main_loop()




