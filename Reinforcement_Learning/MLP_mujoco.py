import argparse
import gym
import os
import sys
import time
from random import randint
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_rl.torch import *
from utils_rl.memory_dataset import *
from utils_rl.store_results import *

from MLPmodel import MLPTrainer, MultiLayerPerceptron, AgentMLP
import csv
from multihead_attention_np import *
from torch.distributions import Normal
from core.common import discounted_rewards, estimate_eta_3, improvement_step_all


torch.set_default_tensor_type(torch.DoubleTensor)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    device = torch.device("cpu")
print('device: ', device)

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')

parser.add_argument('--learn-sigma', default=True, help='update the stddev of the policy')


parser.add_argument('--use-running-state', default=False,
                    help='store running mean and variance instead of states and actions')
parser.add_argument('--max-kl-mlp', type=float, default=0.01, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--num-ensembles', type=int, default=20, metavar='N',
                    help='episode to collect per iteration')
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')

parser.add_argument('--fixed-sigma', default=0.35, type=float, metavar='N',
                    help='sigma of the policy')
parser.add_argument('--epochs-per-iter', type=int, default=40, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--replay-memory-size', type=int, default=30, metavar='G',
                    help='size of training set in episodes ')

parser.add_argument('--h-dim', type=int, default=128, metavar='N',
                    help='dimension of hidden layers in np')

parser.add_argument('--np-batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--early-stopping', type=int, default=-1, metavar='N',
                    help='stop training training when avg_loss reaches it')

parser.add_argument('--v-epochs-per-iter', type=int, default=60, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--v-replay-memory-size', type=int, default=10, metavar='G',
                    help='size of training set in episodes')

parser.add_argument('--v-np-batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--v-early-stopping', type=int, default=-1, metavar='N',
                    help='stop training training when avg_loss reaches it')

parser.add_argument('--directory-path', default='/home/francesco/PycharmProjects/MasterThesis/mujoco learning results/',
                    help='path to plots folder')
parser.add_argument('--tot-steps', default=1000000, type=int,
                    help='total steps in the run')


parser.add_argument('--device-np', default=device,
                    help='device')
parser.add_argument('--dtype', default=torch.float64,
                    help='default type')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')

parser.add_argument('--episode-specific-value', default=False, metavar='N',
                    help='condition the value np on all episodes')
parser.add_argument("--plot-every", type=int, default=1,
                    help='plot every n iter')
args = parser.parse_args()
initial_training = True

args.epochs_per_iter = 20 + 2000 // args.replay_memory_size
args.v_epochs_per_iter = args.epochs_per_iter
args.v_replay_memory_size = args.replay_memory_size

np_spec = '_{},{}rm_{},{}epo_{}h_{}kl_'.format(args.replay_memory_size, args.v_replay_memory_size, args.epochs_per_iter,
                                               args.v_epochs_per_iter, args.h_dim, args.max_kl_mlp)


run_id = '/{}_MLP_{}epi_fixSTD:{}_{}gamma'.format(args.env_name, args.num_ensembles, args.fixed_sigma,
                                                   args.gamma) + np_spec
run_id = run_id.replace('.', ',')
args.directory_path += run_id

mlp_file = args.directory_path + '/{}.csv'.format(args.seed)

#torch.set_default_dtype(args.dtype)

"""environment"""
env = gym.make(args.env_name)
max_episode_len = env._max_episode_steps

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print('state_dim', state_dim)
print('action_dim', action_dim)
is_disc_action = len(env.action_space.shape) == 0

args.fixed_sigma = args.fixed_sigma * torch.ones(action_dim)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)


'''create neural process'''
policy = MultiLayerPerceptron(state_dim, action_dim, args.h_dim).to(args.device_np)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
np_trainer = MLPTrainer(args.device_np, policy, optimizer, print_freq=50)

value_net = MultiLayerPerceptron(state_dim, 1, args.h_dim).to(args.device_np)

value_optimizer = torch.optim.Adam(value_net.parameters(), lr=3e-4)
value_np_trainer = MLPTrainer(args.device_np, value_net, value_optimizer, print_freq=50)

"""create replay memory"""
replay_memory = ReplayMemoryDataset(args.replay_memory_size)
value_replay_memory = ValueReplay(args.v_replay_memory_size)

"""create agent"""
agent = AgentMLP(env, policy, args.num_ensembles, args.device_np, fixed_sigma=args.fixed_sigma)


def train_policy(datasets, epochs=args.epochs_per_iter):
    print('Policy training')
    data_loader = DataLoader(datasets, batch_size=args.np_batch_size, shuffle=True)
    np_trainer.train(data_loader, epochs, early_stopping=None)

def train_value(value_replay_memory):
    print('Value training')
    value_data_loader = DataLoader(value_replay_memory, batch_size=args.v_np_batch_size, shuffle=True)
    value_np_trainer.train(value_data_loader, args.v_epochs_per_iter, early_stopping=None)


def estimate_v_a(complete_dataset, disc_rew):
    ep_rewards = disc_rew
    ep_states = [ep['states'] for ep in complete_dataset]
    real_lens = [ep['real_len'] for ep in complete_dataset]
    estimated_advantages = []

    for states, rewards, real_len in zip(ep_states, ep_rewards, real_lens):
        s_target = states[:real_len, :].unsqueeze(0)
        r_target = rewards.view(1, -1, 1)
        with torch.no_grad():
            values = value_net(s_target)
        advantages = r_target - values
        estimated_advantages.append(advantages.squeeze(0))
    return estimated_advantages


avg_rewards = [0]
tot_steps = [0]

def main_loop():
    colors = []
    num_episodes = args.num_ensembles
    for i in range(num_episodes):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    for i_iter in range(args.max_iter_num):
        batch, log = agent.collect_episodes()
        # generate multiple trajectories that reach the minimum batch_size
        # store_rewards(batch.memory, mlp_file)
        disc_rew_np = discounted_rewards(batch.memory, args.gamma)
        complete_dataset = BaseDataset(batch.memory, disc_rew_np, args.device_np, args.dtype,  max_len=max_episode_len)
        advantages = estimate_v_a(complete_dataset, disc_rew_np)

        improved_context_list_np = improvement_step_all(complete_dataset, advantages, args.max_kl_mlp, args)
        # training
        value_replay_memory.add(complete_dataset)
        train_value(value_replay_memory)

        tn0 = time.time()
        replay_memory.add(complete_dataset)
        train_policy(replay_memory)
        tn1 = time.time()
        tot_steps.append(tot_steps[-1] + log['num_steps'])
        avg_rewards.append(log['avg_reward'])

        if i_iter % args.log_interval == 0:
            print(i_iter)
            print('mlp: \tR_min {:.2f} \tR_max {:.2f} \tR_avg {:.2f}'.format(log['min_reward'], log['max_reward'], log['avg_reward']))
        print('new sigma', args.fixed_sigma)
        store_avg_rewards(tot_steps[-1], avg_rewards[-1], mlp_file.replace(str(args.seed)+'.csv', 'avg'+str(args.seed)+'.csv'))
        if i_iter % args.plot_every == 0:
            plot_rewards_history(tot_steps, avg_rewards)
        if tot_steps[-1] > args.tot_steps:
            break
    """clean up gpu memory"""
    # torch.cuda.empty_cache()



def plot_rewards_history(steps, rews):
    fig_rew, ax_rew = plt.subplots(1, 1)

    ax_rew.plot(steps[1:], rews[1:], c='magenta', label='MLP')
    ax_rew.set_xlabel('number of steps')
    ax_rew.set_ylabel('average reward')
    ax_rew.set_title('Average Reward History')
    plt.legend()
    plt.grid()
    fig_rew.savefig(args.directory_path + run_id.replace('.', ',')+str(args.seed))
    plt.close(fig_rew)


def create_directories(directory_path):
    os.mkdir(directory_path)

try:
    create_directories(args.directory_path)
except FileExistsError:
    pass

main_loop()