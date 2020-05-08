import argparse
import gym
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_rl.torch import *
from utils_rl.memory_dataset import *
from RL_results.store_results import *

from MLPmodel import MLPTrainer, MultiLayerPerceptron, AgentMLP
from utils_rl.env_wrappers import AntWrapper, HumanoidWrapper, InvertedDoublePendulumWrapper
from multihead_attention_np import *
from core.common import discounted_rewards, estimate_v_a, improvement_step_all, critic_estimate, update_critic
from core.mlp_critic import Value


torch.set_default_tensor_type(torch.DoubleTensor)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    device = torch.device("cpu")
print('device: ', device)

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="MountainCarContinuous-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')

parser.add_argument('--learn-sigma', default=True, type=boolean_string, help='update the stddev of the policy')
parser.add_argument('--gae', default=True, type=boolean_string, help='use generalized advantage estimate')
parser.add_argument('--value-net', default=True, type=boolean_string, help='use NN for V estimate')
parser.add_argument('--num-req-steps', type=int, default=1000, metavar='N',
                    help='number of context points to sample from rm')


parser.add_argument('--use-running-state', default=False, type=boolean_string,
                    help='store running mean and variance instead of states and actions')
parser.add_argument('--max-kl-mlp', type=float, default=0.35, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--num-ensembles', type=int, default=5, metavar='N',
                    help='episode to collect per iteration')
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')

parser.add_argument('--fixed-sigma', default=0.45, type=float, metavar='N',
                    help='sigma of the policy')
parser.add_argument('--epochs-per-iter', type=int, default=40, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--replay-memory-size', type=int, default=40, metavar='G',
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
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='discount factor (default: 0.95)')


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

parser.add_argument("--plot-every", type=int, default=1,
                    help='plot every n iter')
args = parser.parse_args()
initial_training = True

args.epochs_per_iter = 2000 // args.replay_memory_size
args.v_epochs_per_iter = args.epochs_per_iter
args.v_replay_memory_size = args.replay_memory_size

np_spec = '_critic:{}_{},{}rm_{},{}epo_{}h_{}kl_'.format(args.value_net, args.replay_memory_size, args.v_replay_memory_size, args.epochs_per_iter,
                                               args.v_epochs_per_iter, args.h_dim, args.max_kl_mlp)


run_id = '/{}_MLP_{}steps_{}epi_fixSTD:{}_{}gamma'.format(args.env_name, args.num_req_steps, args.num_ensembles, args.fixed_sigma,
                                                   args.gamma) + np_spec
run_id = run_id.replace('.', ',')
args.directory_path += run_id

mlp_file = args.directory_path + '/{}.csv'.format(args.seed)

#torch.set_default_dtype(args.dtype)

"""environment"""
if args.env_name == 'Humanoid-v2' or args.env_name == 'HumanoidStandup-v2':
    env = HumanoidWrapper(args.env_name)
elif args.env_name == 'Ant-v2':
    env = AntWrapper(args.env_name)
elif args.env_name == 'InvertedDoublePendulum-v2':
    env = InvertedDoublePendulumWrapper(args.env_name)
else:
    env = gym.make(args.env_name)

max_episode_len = env._max_episode_steps

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print('state_dim', state_dim)
print('action_dim', action_dim)
is_disc_action = len(env.action_space.shape) == 0

if args.fixed_sigma is not None:
    args.fixed_sigma = args.fixed_sigma * torch.ones(action_dim)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)


'''create neural process'''
policy = MultiLayerPerceptron(state_dim, action_dim, args.h_dim).to(args.device_np)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
np_trainer = MLPTrainer(args.device_np, policy, optimizer, print_freq=50)

if args.value_net:
    value_net = Value(state_dim)
    value_net.to(args.device_np)
else:
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


avg_rewards = [0]
tot_steps = [0]

def main_loop():

    for i_iter in range(args.max_iter_num):
        # collect samples
        batch, log = agent.collect_episodes(args.num_req_steps)

        # compute discounted rewards
        disc_rew_mlp = discounted_rewards(batch.memory, args.gamma)
        iter_dataset = BaseDataset(batch.memory, disc_rew_mlp, args.device_np, args.dtype,  max_len=max_episode_len)

        # estimate advantages
        if args.value_net:
            state_list = [ep['states'][:ep['real_len']] for ep in iter_dataset]
            advantages, returns = critic_estimate(value_net, state_list, disc_rew_mlp, args)
            update_critic(value_net, torch.cat(state_list, dim=0), torch.cat(returns, dim=0))
        else:
            advantages = estimate_v_a(iter_dataset, disc_rew_mlp, value_replay_memory, value_net, args)
            value_replay_memory.add(iter_dataset)
            train_value(value_replay_memory)

        # returned context not used but added to iter_dataset inside the function
        improved_context_list = improvement_step_all(iter_dataset, advantages, args.max_kl_mlp, args)

        # training
        replay_memory.add(iter_dataset)
        train_policy(replay_memory)

        # prints & plots
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
    torch.cuda.empty_cache()



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