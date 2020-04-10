import argparse
import gym
import os
import sys
import time
from random import randint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_rl.torch import *
from utils_rl.memory_dataset import *
from utils_rl.store_results import *
from core.agent_samples_all_context import Agent_all_ctxt
#from core.agent_conditionning import Agent_all_ctxt
from core.agent_picker import AgentPicker
from neural_process import NeuralProcess
from training_leave_one_out import NeuralProcessTrainerLoo
from training_module_RL import NeuralProcessTrainerRL
from training_leave_one_out_pick import NeuralProcessTrainerLooPick
import scipy.optimize
from models.mlp_critic import Value
from new_plotting_functions import *
from multihead_attention_np import *
from torch.distributions import Normal
from core.common import discounted_rewards, estimate_v_a, improvement_step_all, compute_gae


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
parser.add_argument('--render', default=False, type=boolean_string,
                    help='render the environment')
parser.add_argument('--mean-action', default=False, type=boolean_string, help='update the stddev of the policy')

parser.add_argument('--learn-sigma', default=False, type=boolean_string, help='update the stddev of the policy')
parser.add_argument('--loo', default=True, type=boolean_string, help='train leaving episode out')
parser.add_argument('--pick', default=True, type=boolean_string, help='choose subset of rm')
parser.add_argument('--rm-as-context', default=True, type=boolean_string, help='choose subset of rm')
parser.add_argument('--gae', default=True, type=boolean_string, help='use generalized advantage estimate')

parser.add_argument('--value-net', default=True, type=boolean_string, help='use NN for V estimate')


parser.add_argument('--num-context', type=int, default=10000, metavar='N',
                    help='number of context points to sample from rm')
parser.add_argument('--num-req-steps', type=int, default=3000, metavar='N',
                    help='number of context points to sample from rm')

parser.add_argument('--use-running-state', default=False, type=boolean_string,
                    help='store running mean and variance instead of states and actions')
parser.add_argument('--max-kl-np', type=float, default=0.69, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--num-ensembles', type=int, default=4, metavar='N',
                    help='episode to collect per iteration')
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='discount factor (default: 0.95)')

parser.add_argument('--fixed-sigma', default=None, type=float, metavar='N',
                    help='sigma of the policy')
parser.add_argument('--epochs-per-iter', type=int, default=20, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--replay-memory-size', type=int, default=35, metavar='G',
                    help='size of training set in episodes ')
parser.add_argument('--z-dim', type=int, default=128, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--r-dim', type=int, default=128, metavar='N',
                    help='dimension of representation space in np')
parser.add_argument('--h-dim', type=int, default=128, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--a-dim', type=int, default=128, metavar='N',
                    help='dimension of representation space in np')
parser.add_argument('--np-batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--early-stopping', type=int, default=-100000, metavar='N',
                    help='stop training training when avg_loss reaches it')

parser.add_argument('--v-epochs-per-iter', type=int, default=20, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--v-replay-memory-size', type=int, default=100, metavar='G',
                    help='size of training set in episodes')
parser.add_argument('--v-z-dim', type=int, default=128, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--v-r-dim', type=int, default=128, metavar='N',
                    help='dimension of representation space in np')

parser.add_argument('--v-np-batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--v-early-stopping', type=int, default=-100000, metavar='N',
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

parser.add_argument('--use-attentive-np', default=False, type=boolean_string, metavar='N',
                     help='use attention in policy and value NPs')
parser.add_argument('--v-use-attentive-np', default=True, type=boolean_string, metavar='N',
                     help='use attention in policy and value NPs')
parser.add_argument('--episode-specific-value', default=False, type=boolean_string, metavar='N',
                    help='condition the value np on all episodes')
parser.add_argument("--plot-every", type=int, default=5,
                    help='plot every n iter')
parser.add_argument("--num-testing-points", type=int, default=1000,
                    help='how many point to use as onl4y testing during NP training')
parser.add_argument('--plot_np_sigma', default=False, type=boolean_string, metavar='N',
                    help='plot stdev')
parser.add_argument("--net-size", type=int, default=1,
                    help='multiplies all net pararms')

args = parser.parse_args()
initial_training = True

args.z_dim *= args.net_size
args.r_dim *= args.net_size
args.h_dim *= args.net_size
args.a_dim *= args.net_size

args.epochs_per_iter = 800 // args.replay_memory_size #+ 10
args.v_epochs_per_iter = args.epochs_per_iter
args.v_replay_memory_size = args.replay_memory_size
args.v_z_dim = args.z_dim
args.v_r_dim = args.r_dim

np_spec = '_NP_critic:{}_{},{}rm_isctxt:{}_{},{}epo_{}z_{}h_{}kl_attention:{}_{}a'.format(args.value_net, args.replay_memory_size, args.v_replay_memory_size,
                                                                args.rm_as_context, args.epochs_per_iter,args.v_epochs_per_iter, args.z_dim,
                                                                args.h_dim, args.max_kl_np, args.use_attentive_np,
                                                                                args.a_dim)


run_id = '/{}_NP_plotNPsigma{}_{}steps_{}epi_fixSTD:{}_{}gamma' \
         '_{}target_loo:{}_picklc:{}_{}context'.format(args.env_name, args.plot_np_sigma, args.num_req_steps, args.num_ensembles,
                                                            args.fixed_sigma,args.gamma, args.num_testing_points, args.loo,
                                                            args.pick, args.num_context) + np_spec
run_id = run_id.replace('.', ',')
args.directory_path += run_id

np_file = args.directory_path + '/{}.csv'.format(args.seed)

#torch.set_default_dtype(args.dtype)

"""environment"""
env = gym.make(args.env_name)
max_episode_len = env._max_episode_steps

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print('state_dim', state_dim)
print('action_dim', action_dim)
is_disc_action = len(env.action_space.shape) == 0

if args.fixed_sigma is not None:
    args.fixed_sigma = args.fixed_sigma * torch.ones(action_dim)
else:
    args.plot_np_sigma = True

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)


'''create neural process'''
if args.use_attentive_np:
    policy_np = AttentiveNeuralProcess(state_dim, action_dim, args.r_dim, args.z_dim, args.h_dim,
                                                       args.a_dim, use_self_att=False).to(args.device_np)
else:
    policy_np = NeuralProcess(state_dim, action_dim, args.r_dim, args.z_dim, args.h_dim).to(args.device_np)
optimizer = torch.optim.Adam(policy_np.parameters(), lr=3e-4)

if args.value_net:
    value_net = Value(state_dim)
    value_net.to(args.device_np)
else:
    if args.v_use_attentive_np:
        value_np = AttentiveNeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_r_dim,
                                          args.a_dim, use_self_att=False).to(args.device_np)
    else:
        value_np = NeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_h_dim).to(args.device_np)
    value_optimizer = torch.optim.Adam(value_np.parameters(), lr=3e-4)
    if args.pick:
        value_np_trainer = NeuralProcessTrainerLooPick(args.device_np, value_np, value_optimizer, pick_dist=None,
                                                       num_context=args.num_context)
    else:
        value_np_trainer = NeuralProcessTrainerLoo(args.device_np, value_np, value_optimizer,
                                                   num_target=args.num_testing_points,
                                                   print_freq=50)
    value_np.training = False

if args.loo and (not args.pick):
    print('using Loo trainer')
    np_trainer = NeuralProcessTrainerLoo(args.device_np, policy_np, optimizer, num_target=args.num_testing_points,
                                         print_freq=50)
if args.loo and args.pick:
    print('using LooPick trainer')
    np_trainer = NeuralProcessTrainerLooPick(args.device_np, policy_np, optimizer, pick_dist=None,
                                            num_context=args.num_context)
if not args.loo:
    print('using RL trainer')
    np_trainer = NeuralProcessTrainerRL(args.device_np, policy_np, optimizer, (1, max_episode_len//2),
                                        print_freq=50)


"""create replay memory"""
replay_memory = ReplayMemoryDataset(args.replay_memory_size)
value_replay_memory = ValueReplay(args.v_replay_memory_size)

"""create agent"""
if not args.pick:
    print('agent all')
    agent_np = Agent_all_ctxt(env, policy_np, args.device_np, running_state=None, render=args.render,
                              attention=args.use_attentive_np, fixed_sigma=args.fixed_sigma, mean_action=args.mean_action)
else:
    print('agent pick')
    agent_np = AgentPicker(env, policy_np, args.device_np, args.num_context, custom_reward=None, pick_dist=None,
                           mean_action=False, render=args.render, running_state=None, fixed_sigma=args.fixed_sigma)

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


def sample_initial_context_normal(num_episodes):
    initial_episodes = []
    #policy_np.apply(init_func)
    if args.fixed_sigma is not None:
        sigma = args.fixed_sigma*0.1
    else:
        sigma = 0.1*ones(action_dim)
    for e in range(2):
        states = torch.zeros([1, max_episode_len, state_dim])

        for i in range(max_episode_len):
            states[:, i, :] = torch.randn(state_dim) #torch.from_numpy(env.observation_space.sample())

        if args.use_attentive_np or True:
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

    policy_np.training = True
    data_loader = DataLoader(train_list, batch_size=args.np_batch_size, shuffle=True)
    np_trainer.train(data_loader, 10*args.epochs_per_iter, early_stopping=0)
    policy_np.training = False

def critic_estimate(states_list, rewards_list, args):
    adv_list = []
    ret_list = []
    for states, rewards in zip(states_list, rewards_list):
        with torch.no_grad():
            values = value_net(states)
        advantages = compute_gae(rewards, values, args.gamma, args.tau)
        returns = values + advantages
        adv_list.append(advantages)
        ret_list.append(returns)
    return adv_list, ret_list


def update_critic(states, returns, l2_reg=1e-3):
    def get_value_loss(flat_params):
        """
        compute the loss for the value network comparing estimated values with empirical ones
        and optimizes the network parameters
        :param flat_params:
        :return:
        """
        set_flat_params_to(value_net, tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()  # MeanSquaredError

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return value_loss.item(), get_flat_grad_from(value_net.parameters()).cpu().numpy()

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                            get_flat_params_from(value_net).detach().cpu().numpy(),
                                                            maxiter=25)
    set_flat_params_to(value_net, tensor(flat_params))


avg_rewards_np = [0]
tot_steps_np = [0]
if args.fixed_sigma is not None:
    sigma_history = [torch.tensor(args.fixed_sigma)]
else:
    sigma_history = []

def main_loop():
    colors = []
    num_episodes = args.num_ensembles
    for i in range(num_episodes):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    improved_context_list_np = sample_initial_context_normal(args.num_ensembles)
    for i_iter in range(args.max_iter_num):

        # generate multiple trajectories that reach the minimum batch_size
        policy_np.training = False
        if len(replay_memory) == 0 or not args.rm_as_context:
            context_list_np = improved_context_list_np
        else:
            context_list_np = replay_memory.data
        ts0 = time.time()
        batch_np, log_np = agent_np.collect_episodes(context_list_np, args.num_req_steps, args.num_ensembles)
        print('sampling:', time.time()-ts0)
        disc_rew_np = discounted_rewards(batch_np.memory, args.gamma)
        iter_dataset_np = BaseDataset(batch_np.memory, disc_rew_np, args.device_np, args.dtype,  max_len=max_episode_len)
        print('np avg actions: ', log_np['action_mean'])
        if args.value_net:
            state_list = [ep['states'][:ep['real_len']] for ep in iter_dataset_np]
            advantages_np, returns = critic_estimate(state_list, disc_rew_np, args)
            update_critic(torch.cat(state_list, dim=0), torch.cat(returns, dim=0))
        else:
            advantages_np = estimate_v_a(iter_dataset_np, disc_rew_np, value_replay_memory, value_np, args)
            value_replay_memory.add(iter_dataset_np)
            train_value_np(value_replay_memory)

        improved_context_list_np = improvement_step_all(iter_dataset_np, advantages_np, args.max_kl_np, args)
        # training
        tn0 = time.time()
        replay_memory.add(iter_dataset_np)
        train_np(replay_memory)
        tn1 = time.time()
        tot_steps_np.append(tot_steps_np[-1] + log_np['num_steps'])
        avg_rewards_np.append(log_np['avg_reward'])
        if i_iter % args.plot_every in [0,1]:
            if 'CartPole' in args.env_name:
                plot_NP_policy_CP(policy_np, replay_memory, i_iter, env, args, use_np_sigma=args.plot_np_sigma)
                plot_rm(replay_memory, i_iter, args)
                plot_improvements_CP(iter_dataset_np, advantages_np, env, i_iter, args, colors)
            elif 'MountainCar' in args.env_name:
                plot_NP_policy_MC(policy_np, replay_memory, i_iter, env, args, use_np_sigma=args.plot_np_sigma)
                plot_improvements_MC(iter_dataset_np, advantages_np, env, i_iter, args, colors)
                plot_improvements_MC_all(iter_dataset_np, advantages_np, env, i_iter, args, colors)

        if i_iter % args.log_interval == 0:
            print(i_iter)
            print('np: \tR_min {:.2f} \tR_max {:.2f} \tR_avg {:.2f}'.format(log_np['min_reward'], log_np['max_reward'], log_np['avg_reward']))
        print('new sigma', args.fixed_sigma)
        plot_rewards_history(tot_steps_np, avg_rewards_np)
        store_avg_rewards(tot_steps_np[-1], avg_rewards_np[-1], np_file.replace(str(args.seed)+'.csv', 'avg'+str(args.seed)+'.csv'))
        if args.fixed_sigma is not None:
            sigma_history.append(torch.tensor(args.fixed_sigma))
        else:
            sigma_history.append(torch.cat([ep['stddevs'] for ep in iter_dataset_np.data]).mean(dim=0))
        plot_sigma_history(sigma_history)
        if tot_steps_np[-1] > args.tot_steps:
            break
        """clean up gpu memory"""
        torch.cuda.empty_cache()



def plot_rewards_history(steps, rews):
    fig_rew, ax_rew = plt.subplots(1, 1)

    ax_rew.plot(steps[1:], rews[1:], c='b', label='AttentiveNP')
    ax_rew.set_xlabel('number of steps')
    ax_rew.set_ylabel('average reward')
    ax_rew.set_title('Average Reward History')
    plt.legend()
    plt.grid()
    fig_rew.savefig(args.directory_path + run_id.replace('.', ',')+str(args.seed))
    plt.close(fig_rew)

def plot_sigma_history(sigma_list):
    colors = ['b', 'r', 'g', 'y']
    fig, ax = plt.subplots(1, 1)
    sigmas = torch.stack(sigma_list, dim=0)
    for i in range(sigmas.shape[-1]):
        ax.scatter(np.arange(len(sigma_list)), sigmas[..., i].cpu(), c=colors[i])
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Standard deviation')
    ax.set_title('Average standard deviation for epsilon =' + str(args.max_kl_np))
    plt.legend()
    plt.grid()
    fig.savefig(args.directory_path + run_id.replace('.', ',')+str(args.seed)+'sigmas')
    plt.close(fig)

def create_plot_4d_gridas(env, args, size=21):
    import gpytorch
    bounds_high = env.observation_space.high
    bounds_low = env.observation_space.low
    num_dim = len(bounds_low)
    xs = []
    nonInf_bounds = []
    for bound_low, bound_high in zip(bounds_low, bounds_high):
        if bound_low < -10e30 or bound_high > 10e30:
            bound_low = -1
            bound_high = 1
        nonInf_bounds.append([bound_low, bound_high])
        xs.append(np.linspace(bound_low, bound_high, size))
    X1, X2, X3, X4 = np.meshgrid(*xs)

    grid = torch.zeros(size, num_dim)
    for i, bounds in enumerate(nonInf_bounds):
        grid_diff = float(bounds[1] - bounds[0]) / (size - 2)
        grid[:, i] = torch.linspace(bounds[0] - grid_diff, bounds[1] + grid_diff, size)

    x = gpytorch.utils.grid.create_data_from_grid(grid)
    x = x.unsqueeze(0).to(args.dtype).to(args.device_np)
    return x, X1, X2, X3, X4, xs




def create_directories(directory_path):
    os.mkdir(directory_path)
    os.mkdir(directory_path + '/Mean improvement/')
    os.mkdir(directory_path + '/policy/')


try:
    create_directories(args.directory_path)
except FileExistsError:
    pass
if __name__ == '__main__':
    main_loop()
