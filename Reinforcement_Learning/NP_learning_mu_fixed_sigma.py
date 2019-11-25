import argparse
import gym
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_rl import *
from new_plotting_functions import *
from core.common import discounted_rewards
from core.agent_ensembles import Agent
from neural_process import NeuralProcess
from training_module_RL import NeuralProcessTrainerRL
from multihead_attention_np import *

from torch.distributions import Normal

# Axes3D import has side effects, it enables using projection='3d' in add_subplot
torch.set_default_tensor_type(torch.DoubleTensor)

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="MountainCarContinuous-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')

parser.add_argument('--improve-mean', default=True,
                    help='whether to use the improved mean or actions sampled from them as context points')
parser.add_argument('--sample-improved-action', default=True,
                    help='sample actions fro improved mean or improve old actions')
parser.add_argument('--use-running-state', default=False,
                    help='store running mean and variance instead of states and actions')
parser.add_argument('--max-kl', type=float, default=5e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--num-ensembles', type=int, default=10, metavar='N',
                    help='episode to collect per iteration')
parser.add_argument('--max-iter-num', type=int, default=501, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')

parser.add_argument('--use-mean', default=False, metavar='N',
                    help='train & condit on improved means/actions'),
parser.add_argument('--fixed-sigma', default=0.05, metavar='N',
                    help='sigma of the policy')
parser.add_argument('--epochs-per-iter', type=int, default=30, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--replay-memory-size', type=int, default=75, metavar='G',
                    help='size of training set in episodes')
parser.add_argument('--z-dim', type=int, default=128, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--r-dim', type=int, default=256, metavar='N',
                    help='dimension of represenation space in np')
parser.add_argument('--h-dim', type=int, default=256, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--np-batch-size', type=int, default=8, metavar='N',
                    help='batch size for np training')

parser.add_argument('--v-epochs-per-iter', type=int, default=30, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--v-replay-memory-size', type=int, default=60, metavar='G',
                    help='size of training set in episodes')
parser.add_argument('--v-z-dim', type=int, default=128, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--v-r-dim', type=int, default=128, metavar='N',
                    help='dimension of represenation space in np')
parser.add_argument('--v-h-dim', type=int, default=128, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--v-np-batch-size', type=int, default=8, metavar='N',
                    help='batch size for np training')

parser.add_argument('--directory-path', default='/cluster/scratch/varolif',
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
parser.add_argument('--episode-specific-value', default=False, metavar='N',
                    help='condition the value np on all episodes')

args = parser.parse_args()

initial_training = True

np_spec = '_{}z_{}rm_{}e_imprM:{}_sampled_a:{}'.format(args.z_dim, args.replay_memory_size,
                                                       args.epochs_per_iter, args.improve_mean,
                                                       args.sample_improved_action)
run_id = '/Value_NP_mean:{}_A:{}_fixSTD:{}_epV:{}_init_tr:{}_{}ep_{}kl_{}gamma_'.format(args.use_mean,
                                                args.use_attentive_np, args.fixed_sigma, args.episode_specific_value,
                                                initial_training, args.num_ensembles, args.max_kl, args.gamma) + np_spec
args.directory_path += run_id

torch.set_default_dtype(args.dtype)

"""environment"""
max_episode_len = 999
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
if args.use_running_state:
    running_state = ZFilter((state_dim,), clip=5)  # running list of states that allows to access precise mean and std
else:
    running_state = None
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

'''create neural process'''
if args.use_attentive_np:
    policy_np = AttentiveNeuralProcess(state_dim, action_dim, args.r_dim, args.z_dim, args.h_dim,
                                                       args.z_dim, use_self_att=True).to(args.device_np)
else:
    policy_np = NeuralProcess(state_dim, action_dim, args.r_dim, args.z_dim, args.h_dim,
                              fixed_sigma=args.fixed_sigma).to(args.device_np)

optimizer = torch.optim.Adam(policy_np.parameters(), lr=3e-4)
np_trainer = NeuralProcessTrainerRL(args.device_np, policy_np, optimizer,
                                    num_context_range=(400, 500),
                                    num_extra_target_range=(400, 500),
                                    print_freq=50)

if args.use_attentive_np:
    value_np = AttentiveNeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_h_dim,
                                      args.z_dim, use_self_att=True).to(args.device_np)
else:
    value_np = NeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_h_dim).to(args.device_np)
value_optimizer = torch.optim.Adam(value_np.parameters(), lr=3e-4)
value_np_trainer = NeuralProcessTrainerRL(args.device_np, value_np, value_optimizer,
                                          num_context_range=(400, 500),
                                          num_extra_target_range=(400, 500),
                                          print_freq=50)
"""create replay memory"""
replay_memory = ReplayMemoryDataset(args.replay_memory_size)
value_replay_memory = ValueReplay(args.v_replay_memory_size)

"""create agent"""
agent = Agent(env, policy_np, args.device_np, running_state=running_state, render=args.render, attention=args.use_attentive_np)


def estimate_eta_2(actions, means, stddevs, disc_rews):
    d = actions.shape[-1]
    if d > 1:
        raise NotImplementedError('compute eta not implemented for action space of dim>1')
    else:
        iter_sum = 0
        eps = tensor(args.max_kl).to(args.dtype)
        T = tensor(actions.shape[0]).to(args.dtype)
        for action, mean, stddev, disc_reward in zip(actions, means, stddevs, disc_rews):
            iter_sum += ((disc_reward ** 2) * (action - mean) ** 2) / stddev ** 4
        denominator = iter_sum.to(args.dtype)
        return torch.sqrt((T*eps)/denominator)


def improvement_step(complete_dataset, estimated_disc_rew, values_stdevs):
    all_improved_context = []
    for episode, disc_rewards, values_std in zip(complete_dataset, estimated_disc_rew, values_stdevs):
        real_len = episode['real_len']
        states = episode['states'][:real_len]
        actions = episode['actions'][:real_len]
        means = episode['means'][:real_len]
        stddevs = episode['stddevs'][:real_len]
        eta = estimate_eta_2(actions, means, stddevs, disc_rewards)
        new_padded_actions = torch.zeros_like(episode['actions'])
        new_padded_means = torch.zeros_like(episode['means'])
        i = 0
        for state, action, mean, stddev, disc_reward, value_std in zip(states, actions, means, stddevs, disc_rewards, values_std):
            new_mean = mean + eta * disc_reward * ((action - mean) / stddev)
            distr = Normal(new_mean, stddev)
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
    np_trainer.train(data_loader, epochs, early_stopping=-2000)


def train_value_np(value_replay_memory):
    #print('Value training')
    value_np.training = True
    value_data_loader = DataLoader(value_replay_memory, batch_size=args.np_batch_size, shuffle=True)
    value_np_trainer.train(value_data_loader, args.v_epochs_per_iter, early_stopping=100)
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
    plot_NP_value(value_np, all_states, all_values, all_episodes, all_rewards, env, args, i_iter)
    return estimated_disc_rew, value_stddevs

def sample_initial_context_uniform(num_episodes):
    initial_episodes = []
    bounds_high = env.action_space.high
    bounds_low = env.action_space.low
    action_delta = (bounds_high-bounds_low)/num_episodes/2
    start = (bounds_low + action_delta).item()
    end = (bounds_high - action_delta).item()
    initial_means = torch.linspace(start, end, num_episodes)
    if args.fixed_sigma is not None:
        std = args.fixed_sigma
    else:
        std = action_delta.item()
    for e in range(num_episodes):
        states = torch.zeros([1, max_episode_len, state_dim])
        actions = torch.zeros([1, max_episode_len, action_dim])
        for i in range(max_episode_len):
            states[:, i, :] = torch.from_numpy(env.observation_space.sample())
            actions[:, i, :] = Normal(initial_means[e], std).sample()

        initial_episodes.append([states, actions, max_episode_len])
    return initial_episodes

def sample_initial_context_normal(num_episodes):
    initial_episodes = []
    means_variance_gain = 1/100
    z_init = Normal(0, 1/means_variance_gain)
    for e in range(num_episodes):
        states = torch.zeros([1, max_episode_len, state_dim])
        for i in range(max_episode_len):
            states[:, i, :] = torch.from_numpy(env.observation_space.sample())

        z_sample = z_init.sample([1, args.z_dim])
        z_sample = z_sample.unsqueeze(1).repeat(1, max_episode_len, 1)
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


def main_loop():
    colors = []
    num_episodes = args.num_ensembles
    for i in range(num_episodes):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    #print('sampling initial context')
    improved_context_list = sample_initial_context_uniform(args.num_ensembles)
    plot_initial_context(improved_context_list, colors, env, args, '00')
    if initial_training:
        train_on_initial(improved_context_list)
    avg_rewards = []
    for i_iter in range(args.max_iter_num):
        #print('sampling episodes')
        # (1)
        # generate multiple trajectories that reach the minimum batch_size
        # introduce param context=None when np is policy, these will be the context points used to predict
        policy_np.training = False
        batch, log = agent.collect_episodes(improved_context_list)  # batch of batch_size transitions from multiple
        #print(log['num_steps'], log['num_episodes'])                # episodes (separated by mask=0). Stored in Memory

        disc_rew = discounted_rewards(batch.memory, args.gamma)
        complete_dataset = BaseDataset(batch.memory, disc_rew, args.device_np, args.dtype)

        value_replay_memory.add(complete_dataset)
        train_value_np(value_replay_memory)
        estimated_disc_rew, values_stdevs = estimate_disc_rew(complete_dataset, i_iter, episode_specific_value=args.episode_specific_value)

        t0 = time.time()
        improved_context_list = improvement_step(complete_dataset, estimated_disc_rew, values_stdevs)
        t1 = time.time()
        #plot_initial_context(improved_context_list, colors, env, args, i_iter)
        # plot improved context and actions' discounted rewards
        plot_improvements(complete_dataset, estimated_disc_rew, env, i_iter, args, colors)

        # create training set
        replay_memory.add(complete_dataset)
        train_np(replay_memory)
        #plot_training_set(i_iter, replay_memory, env, args)

        plot_NP_policy(policy_np, improved_context_list, i_iter, log['avg_reward'], env, args, colors)

        avg_rewards.append(log['avg_reward'])
        if i_iter % args.log_interval == 0 and False:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                  i_iter, log['sample_time'], t1 - t0, log['min_reward'], log['max_reward'], log['avg_reward']))

    plot_rewards_history(avg_rewards, args)

    """clean up gpu memory"""
    torch.cuda.empty_cache()


create_directories(args.directory_path)
main_loop()


