import argparse
import gym
import os
import sys
import time
from random import randint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import gpytorch
from utils import context_target_split
from plotting_functions_DKL import plot_posterior
from torch.utils.data import DataLoader
from core.agent_ensembles_all_context import Agent
from MeanInterpolatorModel import MeanInterpolator, MITrainer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_rl import *
from new_plotting_functions import plot_initial_context, plot_rewards_history, set_labels, create_plot_grid
from core.common import discounted_rewards
from core.agent_picker import AgentPicker

from multihead_attention_np import *
from torch.distributions import Normal
from weights_init import InitFunc

torch.set_default_tensor_type(torch.DoubleTensor)
if torch.cuda.is_available() and False:
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    device = torch.device("cpu")
print('device: ', device)

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="CartPole-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')

parser.add_argument('--use-running-state', default=False,
                    help='store running mean and variance instead of states and actions')
parser.add_argument('--max-kl', type=float, default=0.1, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--num-ensembles', type=int, default=20, metavar='N',
                    help='episode to collect per iteration')
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')

parser.add_argument('--z-dim', type=int, default=4, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--h-dim', type=int, default=256, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--fixed-sigma', default=0.75, metavar='N', type=float,
                    help='sigma of the policy')
parser.add_argument('--epochs-per-iter', type=int, default=60, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--replay-memory-size', type=int, default=50, metavar='G',
                    help='size of training set in episodes ')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--scaling', default=None, metavar='N',
                    help='feature extractor scaling')

parser.add_argument('--num-context', type=int, default=1000, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument("--pick", type=bool, default=False,
                    help='whether to select a subst of context points')
parser.add_argument("--loo", type=bool, default=False,
                    help='plot every n iter')

parser.add_argument("--lr_nn", type=float, default=1e-3,
                    help='plot every n iter')

parser.add_argument('--directory-path', default='/home/francesco/PycharmProjects/MasterThesis/MI learning results/',
                    help='path to plots folder')
parser.add_argument('--device-np', default=device,
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

parser.add_argument("--plot-every", type=int, default=1,
                    help='plot every n iter')
args = parser.parse_args()
initial_training = True
init_func = InitFunc.init_zero

max_episode_len = 200

dkl_spec = 'MI_{}epi_{}epo_{}rm_{}lr_nn_{}z_{}h_{}_/'.format(args.num_ensembles, args.epochs_per_iter, args.replay_memory_size,
                                                            args.lr_nn, args.z_dim, args.h_dim, args.scaling)
run_id = 'CARTPOLE_fixSTD:{}_{}ep_{}kl_{}gamma_pick{}_{}ctx_loo{}_no_z'.format(args.fixed_sigma, args.num_ensembles, args.max_kl,
                                                             args.gamma, args.pick, args.num_context, args.loo) + dkl_spec
args.directory_path += run_id

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

#torch.set_default_dtype(args.dtype)
def sample_initial_context_normal(num_ensembles):
    initial_episodes = []
    for e in range(num_ensembles):
        sigma = args.fixed_sigma
        if sigma is None:
            sigma = 0.2

        states = torch.zeros([1, max_episode_len, state_dim])
        for i in range(max_episode_len):
            states[:, i, :] = torch.randn(state_dim)  # torch.from_numpy(env.observation_space.sample())
        actions_init = Normal(torch.zeros([1, max_episode_len, action_dim]),
                              sigma*torch.ones([1, max_episode_len, action_dim])).sample()
        initial_episodes.append([states, actions_init, max_episode_len])
    return initial_episodes

'''create policy model'''
improved_context_list = sample_initial_context_normal(args.num_ensembles)

model = MeanInterpolator(state_dim, args.h_dim, args.z_dim, scaling=args.scaling).to(device).double()



optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters(), 'lr':args.lr_nn},
    {'params': model.interpolator.parameters(), 'lr':args.lr_nn}])

# train
if args.loo:
    raise  NotImplementedError
else:
    model_trainer = MITrainer(device, model, optimizer, args, print_freq=30)

"""create replay memory"""
replay_memory = ReplayMemoryDataset(args.replay_memory_size, use_mean=True)

"""create agent"""
if args.pick:
    agent = AgentPicker(env, model, args.device_np, args.num_context, running_state=running_state, render=args.render,
                        pick_dist=None, fixed_sigma=args.fixed_sigma)
else:

    agent = Agent(env, model, args.device_np, custom_reward=None, attention=False, mean_action=False, render=args.render,
                  running_state=running_state, fixed_sigma=args.fixed_sigma)

def estimate_eta_3(actions, means, advantages, sigmas):
    """Compute learning step from all the samples of previous iteration"""
    d = actions.shape[-1]
    if d > 1:
        raise NotImplementedError('compute eta not implemented for action space of dim>1')
    else:
        stddev = args.fixed_sigma
        iter_sum = 0
        eps = tensor(args.max_kl).to(args.dtype)
        T = tensor(actions.shape[0]).to(args.dtype)
        for action, mean, disc_reward, sigma in zip(actions, means, advantages, sigmas):
            if stddev is None:
                stddev = sigma
            iter_sum += ((disc_reward ** 2) * (action - mean) ** 2) / (2 * (stddev ** 4))
        denominator = iter_sum.to(args.dtype)
        return torch.sqrt((T * eps) / denominator)


def improvement_step_all(complete_dataset, estimated_adv):
    """Perform improvement step using same eta for all episodes"""
    all_improved_context = []
    with torch.no_grad():
        all_states, all_means, all_stdv, all_actions = merge_padded_lists([episode['states'] for episode in complete_dataset],
                                                                [episode['means'] for episode in complete_dataset],
                                                                [episode['stddevs'] for episode in complete_dataset],
                                                                [episode['actions'] for episode in complete_dataset],
                                                                 max_lens=[episode['real_len'] for episode in complete_dataset])
        all_advantages = [adv for ep in estimated_adv for adv in ep]
        eta = estimate_eta_3(all_actions, all_means, all_advantages, all_stdv)
        for episode, episode_adv in zip(complete_dataset, estimated_adv):
            real_len = episode['real_len']
            states = episode['states'][:real_len]
            actions = episode['actions'][:real_len]
            means = episode['means'][:real_len]
            new_padded_actions = torch.zeros_like(episode['actions'])
            new_padded_means = torch.zeros_like(episode['means'])
            i = 0
            for state, action, mean, advantage, stddev in zip(states, actions, means, episode_adv, all_stdv):
                if args.fixed_sigma is None:
                    sigma = stddev
                else:
                    sigma = args.fixed_sigma
                new_mean = mean + eta * advantage * ((action - mean) / sigma)
                distr = Normal(new_mean, sigma)
                new_action = distr.sample()
                new_padded_actions[i, :] = new_action
                new_padded_means[i, :] = new_mean
                i += 1
            episode['new_means'] = new_padded_means
            episode['new_actions'] = new_padded_actions

            all_improved_context.append([episode['states'].unsqueeze(0), new_padded_means.unsqueeze(0), real_len])

    return all_improved_context


def train_np(datasets, epochs=args.epochs_per_iter):
    print('Policy training')
    data_loader = DataLoader(datasets, batch_size=args.batch_size, shuffle=True)
    model_trainer.train_rl(data_loader, args.epochs_per_iter, early_stopping=None)


def estimate_disc_rew(all_episodes, i_iter, episode_specific_value=False):
    value_np.training = False
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
                estimated_disc_rew.append(r_est.view(-1).cpu().numpy())
                value_stddevs.append(values_distr.stddev.view(-1).cpu().numpy())
            all_values.append(values)
    #if i_iter % args.plot_every == 0:
    #    plot_NP_value(value_np, all_states, all_values, all_episodes, all_rewards, value_replay_memory, env, args, i_iter)
    return estimated_disc_rew, value_stddevs



def create_directories(directory_path):

    os.mkdir(directory_path)
    os.mkdir(directory_path + '/Mean improvement/')
    os.mkdir(directory_path + '/z/')

avg_rewards = []


def main_loop(improved_context_list):
    colors = []
    num_episodes = args.num_ensembles
    for i in range(num_episodes):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    for i_iter in range(args.max_iter_num):
        print('sampling episodes')        # (1)
        # generate multiple trajectories that reach the minimum batch_size
        model.eval()
        batch, log = agent.collect_episodes(improved_context_list, render=(i_iter%10==0))

        disc_rew = discounted_rewards(batch.memory, args.gamma)
        complete_dataset = BaseDataset(batch.memory, disc_rew, args.device_np, args.dtype,  max_len=max_episode_len)
        #advantages, values = estimate_v_a(disc_rew, )

        t0 = time.time()
        improved_context_list = improvement_step_all(complete_dataset, disc_rew)
        t1 = time.time()

        # create training set
        tn0 = time.time()
        replay_memory.add(complete_dataset)
        train_np(replay_memory)
        tn1 = time.time()

        tv0 = time.time()
        if i_iter % args.plot_every == 0:
            # plot_initial_context(improved_context_list, colors, env, args, i_iter)
            # plot_training_set(i_iter, replay_memory, env, args)
            plot_policy(model, improved_context_list, replay_memory, i_iter, log['avg_reward'], env, args, colors)
            plot_improvements(complete_dataset, disc_rew, env, i_iter, args, colors)
        tv1 = time.time()

        avg_rewards.append(log['avg_reward'])
        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f} \tT_update {:.4f} \tR_min {:.2f} \tR_max {:.2f} \tR_avg {:.2f}'.format(
                  i_iter, log['sample_time'], t1 - t0, log['min_reward'], log['max_reward'], log['avg_reward']))
            print('Training:  \tT_policy {:.2f}  \nT_plots {:.2f}'.format(tn1-tn0, tv1-tv0))
        if log['avg_reward'] > 195:
            print('converged')
            plot_rewards_history(avg_rewards, args)
        if i_iter % args.plot_every == 0:
            plot_rewards_history(avg_rewards, args)
    plot_rewards_history(avg_rewards, args)

    """clean up gpu memory"""
    torch.cuda.empty_cache()

def create_plot_4d_grid(env, args, size=20):
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

def plot_policy(model, all_context_xy, rm, iter_pred, avg_rew, env, args, colors):
    size = 10
    fig = plt.figure(figsize=(16,8))
    x, X1, X2, X3, X4, xs = create_plot_4d_grid(env, args, size=size)
    mu_list = []
    stds_list = []
    xp1, xp2 = np.meshgrid(xs[0], xs[2])
    xp3, xp4 = np.meshgrid(xs[1], xs[3])
    middle_vel = len(X2) // 2

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 3, projection='3d')
    ax1c = fig.add_subplot(2,2,2, projection='3d')
    ax2c = fig.add_subplot(2,2,4, projection='3d')

    bounds_high = env.observation_space.high
    bounds_low = env.observation_space.low
    x1 = np.linspace(bounds_low[0], bounds_high[0], 10)
    x3 = np.linspace(bounds_low[2], bounds_high[2], 10)
    x2 = np.linspace(-1, 1, 10)
    x4 = np.linspace(-1, 1, 10)
    X1, X3 = np.meshgrid(x1, x3)
    X2, X4 = np.meshgrid(x2, x4)
    middle_vel = len(X2) // 2
    x_context, y_context = merge_context(all_context_xy)
    z_mean = torch.zeros([10,10,10,10])
    with torch.no_grad():
        for e1, _x1 in enumerate(x1):
            for e2, _x2 in enumerate(x2):
                for e3, _x3 in enumerate(x3):
                    for e4, _x4 in enumerate(x4):
                        state = tensor([_x1, _x2, _x3, _x4], device=device).unsqueeze(0)
                        z_mean[e1, e2, e3, e4] = model(x_context, y_context, state).item()
    ax1.plot_surface(X1, X3, z_mean[:, middle_vel, :, middle_vel].numpy(), cmap='viridis', vmin=-1., vmax=1.)
    ax2.plot_surface(X2, X4, z_mean[middle_vel, :, middle_vel, :].numpy(), cmap='viridis', vmin=-1., vmax=1.)

    for e, context_xy in enumerate(all_context_xy):
        x_context, y_context, real_len = context_xy
        ax1c.scatter(x_context[0, :, [0]].cpu(), x_context[0, :, [2]].cpu(), y_context.view(-1, 1).cpu(), cmap='viridis', vmin=-1.,
                     vmax=1.)
        ax2c.scatter(x_context[0, :, [1]].cpu(), x_context[0, :, [3]].cpu(), y_context.view(-1, 1).cpu(), cmap='viridis', vmin=-1.,
                     vmax=1.)


    fig.suptitle('DKL policy for iteration {}, avg rew {} '.format(iter_pred, int(avg_rew)), fontsize=20)
    ax1.set_title('cart v: {:.2f}, bar v:{:.2f}'.format(xs[1][middle_vel], xs[3][middle_vel]))
    ax1.set_xlabel('cart position')
    ax1.set_ylabel('bar angle')
    ax1.set_zlabel('action')
    ax1.set_zlim(-1, 1)
    ax2.set_title('cart p: {:.2f}, bar angle:{:.2f}'.format(xs[0][middle_vel], xs[2][middle_vel]))
    ax2.set_xlabel('cart velocity')
    ax2.set_ylabel('bar velocity')
    ax2.set_zlabel('action')
    ax2.set_zlim(-1, 1)
    ax1c.set_title('context points')
    ax1c.set_xlabel('cart position')
    ax1c.set_ylabel('bar angle')
    ax1c.set_zlabel('action')
    ax1c.set_zlim(-1, 1)
    ax2c.set_title('context points')
    ax2c.set_xlabel('cart velocity')
    ax2c.set_ylabel('bar velocity')
    ax2c.set_zlabel('action')
    ax2c.set_zlim(-1, 1)


    fig.savefig(args.directory_path +str(iter_pred), dpi=250)

    plt.close(fig)

def set_bounds(axes, dims):
    bounds_high = env.observation_space.high
    bounds_low = env.observation_space.low
    num_dim = len(bounds_low)
    nonInf_bounds = []
    for bound_low, bound_high in zip(bounds_low, bounds_high):
        if bound_low < -10e30 or bound_high > 10e30:
            bound_low = -1
            bound_high = 1
        nonInf_bounds.append([bound_low, bound_high])
    for ax in axes:
        ax.set_xlim(nonInf_bounds[dims[0]])
        ax.set_ylim(nonInf_bounds[dims[1]])

def plot_improvements(all_dataset, est_rewards, env, i_iter, args, colors):

    name = 'Improvement iter ' + str(i_iter)
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(name, fontsize=20)
    ax = fig.add_subplot(121, projection='3d')
    name_c = 'Context improvement iter ' + str(i_iter)
    ax.set_title(name_c)
    ax_rew = fig.add_subplot(122, projection='3d')
    set_bounds([ax, ax_rew], [0,2])
    for a in [ax, ax_rew]:
        a.set_zlim(-1, 1)
        a.set_xlabel('cart position')
        a.set_ylabel('bar angle')
    for e, episode in enumerate(all_dataset):
        real_len = episode['real_len']
        states = episode['states'][:real_len].cpu()
        disc_rew = episode['discounted_rewards'][:real_len].cpu()
        actions = episode['actions'][:real_len].cpu()
        means = episode['means'][:real_len].cpu()
        new_means = episode['new_means'][:real_len].cpu()
        est_rew = est_rewards[e]
        if e == 0:
            ax.scatter(states[:, 0].numpy(), states[:, 2].numpy(), means[:, 0].numpy(), c='k', label='sampled', alpha=0.3)
            ax.scatter(states[:, 0].numpy(), states[:, 2].numpy(), new_means[:, 0].numpy(), c=new_means[:, 0].numpy(), marker='+', label='improved', alpha=0.6)
            leg = ax.legend(loc="upper right")
        else:
            ax.scatter(states[:, 0].numpy(), states[:, 2].numpy(), means[:, 0].numpy(), c='k', alpha=0.3)
            ax.scatter(states[:, 0].numpy(), states[:, 2].numpy(), new_means[:, 0].numpy(), c=new_means[:, 0].numpy(), marker='+', alpha=0.6)

        a = ax_rew.scatter(states[:, 0].numpy(), states[:, 2].numpy(), actions[:, 0].numpy(), c=est_rew[:], cmap='viridis', alpha=0.5)

    cb = fig.colorbar(a)
    cb.set_label('Discounted rewards')
    ax_rew.set_title('Discounted rewards')
    fig.savefig(args.directory_path+'/Mean improvement/'+name, dpi=250)
    plt.close(fig)




create_directories(args.directory_path)
main_loop(improved_context_list)
