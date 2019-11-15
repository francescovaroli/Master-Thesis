import argparse
import gym
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_rl import *
from plotting_functions import *
from core.common import discounted_rewards
from core.agent_np import Agent
from neural_process import NeuralProcess
from training_module_RL import NeuralProcessTrainerRL

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
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--min-batch-size', type=int, default=5*999, metavar='N',
                    help='minimal batch size per TRPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=501, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--epochs-per-iter', type=int, default=30, metavar='G',
                    help='training epochs of NP')

parser.add_argument('--replay-memory-size', type=int, default=15, metavar='G',
                    help='size of training set in episodes')
parser.add_argument('--z-dim', type=int, default=128, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--r-dim', type=int, default=256, metavar='N',
                    help='dimension of represenation space in np')
parser.add_argument('--h-dim', type=int, default=256, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--np-batch-size', type=int, default=8, metavar='N',
                    help='batch size for np training')

parser.add_argument('--v-replay-memory-size', type=int, default=80, metavar='G',
                    help='size of training set in episodes')
parser.add_argument('--v-z-dim', type=int, default=128, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--v-r-dim', type=int, default=256, metavar='N',
                    help='dimension of represenation space in np')
parser.add_argument('--v-h-dim', type=int, default=256, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--v-np-batch-size', type=int, default=8, metavar='N',
                    help='batch size for np training')

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

args = parser.parse_args()


frac_replace_actions = 1


np_spec = 'ValueNP_{}z_{}rm_{}e_imprM:{}_sampled_a:{}_RS:{}_replace{}'.format(args.z_dim, args.replay_memory_size, args.epochs_per_iter, args.improve_mean,
                                                                      args.sample_improved_action, args.use_running_state, frac_replace_actions)
run_id = '{}b_{}kl_{}gamma_'.format(args.min_batch_size, args.max_kl, args.gamma) + np_spec
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
policy_np = NeuralProcess(state_dim, action_dim, args.r_dim, args.z_dim, args.h_dim).to(args.device_np)
optimizer = torch.optim.Adam(policy_np.parameters(), lr=3e-4)
np_trainer = NeuralProcessTrainerRL(args.device_np, policy_np, optimizer,
                                    num_context_range=(400, 500),
                                    num_extra_target_range=(400, 500),
                                    print_freq=100)

value_np = NeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_h_dim).to(args.device_np)
value_optimizer = torch.optim.Adam(value_np.parameters(), lr=3e-4)
value_np_trainer = NeuralProcessTrainerRL(args.device_np, value_np, value_optimizer,
                                          num_context_range=(400, 500),
                                          num_extra_target_range=(400, 500),
                                          print_freq=100)
"""create replay memory"""
replay_memory = ReplayMemoryDataset(args.replay_memory_size)
value_replay_memory = ValueReplay(args.v_replay_memory_size)

"""create agent"""
agent = Agent(env, policy_np, args.device_np, running_state=running_state, render=args.render, num_threads=args.num_threads)


def estimate_eta(episode):
    d = episode[0].action.shape[0]
    if d > 1:
        raise NotImplementedError('compute eta not implemented for action space of dim>1')
    else:
        iter_sum = 0
        eps = tensor(args.max_kl).to(args.dtype)
        T = tensor(len(episode)).to(args.dtype)
        for t in episode:
            iter_sum += ((t.disc_rew ** 2) * (t.action - t.mean) ** 2) / t.stddev ** 4
        denominator = torch.from_numpy(iter_sum).to(args.dtype)
        return torch.sqrt((T*eps)/denominator)


def improvement_step(memory):
    print('improving actions')
    first = True
    for episode in memory:
        eta = estimate_eta(episode)
        for transition in episode:
            state = torch.from_numpy(transition.state).to(args.dtype)
            action = torch.from_numpy(transition.action).to(args.dtype)
            mean = torch.from_numpy(transition.mean).to(args.dtype)
            stddev = torch.from_numpy(transition.stddev).to(args.dtype)
            discounted_reward = tensor(transition.disc_rew).to(args.dtype).unsqueeze(0)

            new_mean = mean + eta*discounted_reward*((action-mean)/stddev)
            if args.sample_improved_action:
                distr = Normal(new_mean[0], stddev[0])
                new_action = distr.sample()
            else:
                new_action = action + eta*discounted_reward*((action-mean)/stddev)
                new_action = new_action[0]

            if first:
                all_new_states = state.unsqueeze(0)
                all_new_actions = new_action
                all_new_means = new_mean[0]
                first = False
            else:
                all_new_states = torch.cat((all_new_states, state.unsqueeze(0)), dim=0)
                all_new_means = torch.cat((all_new_means, new_mean[0]), dim=0)
                all_new_actions = torch.cat((all_new_actions, new_action), dim=0)


    return {'states':all_new_states.unsqueeze(0), 'means':all_new_means.unsqueeze(0), 'actions':all_new_actions.unsqueeze(0)}


def sample_context(x, y, num_context=100):
    x = x.to(args.dtype).to(args.device_np)
    y = y.to(args.dtype).to(args.device_np)
    num_points = x.shape[1]
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_context,
                                 replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    return x_context, y_context


def train_np(datasets):
    policy_np.training = True
    data_loader = DataLoader(datasets, batch_size=args.np_batch_size, shuffle=True)
    np_trainer.train(data_loader, args.epochs_per_iter, early_stopping=100)


def train_value_np(value_replay_memory):
    value_np.training = True
    value_data_loader = DataLoader(value_replay_memory, batch_size=args.np_batch_size, shuffle=True)
    value_np_trainer.train(value_data_loader, args.epochs_per_iter, early_stopping=100)
    value_np.training = False


def estimate_disc_rew(all_episodes, i_iter):
    estimated_disc_rew = []
    for episode in all_episodes.data:
        real_len = episode['real_len']
        x = episode['states'][:real_len].unsqueeze(0)
        context_y = episode['discounted_rewards'][:real_len].unsqueeze(0)
        with torch.no_grad():
            values_distr = value_np(x, context_y, x)
            values = values_distr.mean
            r_est = context_y - values
            estimated_disc_rew.append(r_est.view(-1).numpy())

    plot_NP_value(value_np, x, context_y, values, r_est, env, args, i_iter)
    return estimated_disc_rew


def sample_initial_context():
    print('sampling initial context')
    def fix(tensor):
        return tensor.to(args.dtype).unsqueeze(0)
    all_states = fix(torch.from_numpy(env.observation_space.sample()))
    all_actions = fix(torch.from_numpy(env.action_space.sample()))
    for i in range(args.min_batch_size-1):
        state = fix(torch.from_numpy(env.observation_space.sample()))
        action = fix(torch.from_numpy(env.action_space.sample()))
        all_states = torch.cat((all_states, state), dim=0)
        all_actions = torch.cat((all_actions, action), dim=0)

    return [all_states.unsqueeze(0), all_actions.unsqueeze(0)]


def create_directories(directory_path):
    os.mkdir(directory_path)
    os.mkdir(directory_path + '/policy/')
    os.mkdir(directory_path + '/value/')
    os.mkdir(directory_path + '/policy/'+'/NP estimate/')
    os.mkdir(directory_path + '/policy/' + '/Mean improvement/')
    os.mkdir(directory_path + '/policy/' + '/Training/')


def main_loop():
    improved_context = sample_initial_context()
    avg_rewards = []
    for i_iter in range(args.max_iter_num):
        print('sampling episodes')
        # (1)
        # generate multiple trajectories that reach the minimum batch_size
        # introduce param context=None when np is policy, these will be the context points used to predict
        policy_np.training = False
        batch, log, memory = agent.collect_samples(args.min_batch_size, context=improved_context)  # batch of batch_size transitions from multiple
        print(log['num_steps'], log['num_episodes'])                     # episodes (separated by mask=0). Stored in Memory

        disc_rew = discounted_rewards(batch, args.gamma)
        complete_dataset = BaseDataset(batch, disc_rew, args.device_np, args.dtype)
        value_replay_memory.add(complete_dataset)
        train_value_np(value_replay_memory)

        estimated_disc_rew = estimate_disc_rew(complete_dataset, i_iter)
        memory.set_disc_rew(estimated_disc_rew)

        t0 = time.time()
        all_improved_context = improvement_step(batch)
        t1 = time.time()
        key = 'means' if args.improve_mean else 'actions'
        improved_context = [all_improved_context['states'], all_improved_context[key]]

        # plot improved context and actions' discounted rewards
        plot_improvements(batch, improved_context, env, i_iter, args)

        # create training set
        training_set = all_improved_context['means']
        frac_action_in_training = int(frac_replace_actions * training_set.shape[1])
        training_set[:, :frac_action_in_training, :] = all_improved_context['actions'][:, :frac_action_in_training, :]

        dataset = MemoryDatasetNP(batch, training_set, args.device_np, args.dtype, max_len=999)
        replay_memory.add(dataset)

        plot_training_set(i_iter, replay_memory, env, args)

        print('replay memory size:', len(replay_memory))
        train_np(replay_memory)

        plot_NP_policy(policy_np, improved_context, i_iter, log['avg_reward'], env, args, num_samples=1)

        avg_rewards.append(log['avg_reward'])
        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                  i_iter, log['sample_time'], t1 - t0, log['min_reward'], log['max_reward'], log['avg_reward']))

    plot_rewards_history(avg_rewards, args)

    """clean up gpu memory"""
    torch.cuda.empty_cache()

create_directories(args.directory_path)
main_loop()
