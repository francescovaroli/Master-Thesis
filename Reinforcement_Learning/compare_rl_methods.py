import argparse
import gym
import os
import sys
import time
from random import randint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
from core.agent_ensembles_all_context import Agent_all_ctxt
from MeanInterpolatorModel import MeanInterpolator, MITrainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_rl import *
from new_plotting_functions import plot_initial_context, plot_rewards_history, set_labels, create_plot_grid
from core.common import discounted_rewards
from core.agent_picker import AgentPicker

from multihead_attention_np import *
from torch.distributions import Normal
from weights_init import InitFunc
import argparse
import gym
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gpytorch
from utils_rl import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from core.trpo import trpo_step
from core.common import estimate_advantages
from core.agent import Agent

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

parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl-trpo', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
parser.add_argument('--num-threads', type=int, default=5, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--min-batch-size', type=int, default=3994, metavar='N',
                    help='minimal batch size per TRPO update (default: 2048)')

parser.add_argument('--use-running-state', default=False,
                    help='store running mean and variance instead of states and actions')
parser.add_argument('--max-kl', type=float, default=0.4, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--num-ensembles', type=int, default=10, metavar='N',
                    help='episode to collect per iteration')
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')

parser.add_argument('--z-dim', type=int, default=8, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--h-dim', type=int, default=50, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--fixed-sigma', default=0.05, metavar='N', type=float,
                    help='sigma of the policy')
parser.add_argument('--epochs-per-iter', type=int, default=60, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--replay-memory-size', type=int, default=50, metavar='G',
                    help='size of training set in episodes ')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--scaling', default='uniform', metavar='N',
                    help='feature extractor scaling')

parser.add_argument('--num-context', type=int, default=1000, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument("--pick", type=bool, default=False,
                    help='whether to select a subst of context points')
parser.add_argument("--loo", type=bool, default=True,
                    help='plot every n iter')

parser.add_argument("--lr_nn", type=float, default=1e-3,
                    help='plot every n iter')

parser.add_argument('--directory-path', default='/home/francesco/PycharmProjects/MasterThesis/comparison/',
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

dkl_spec = 'MI_{}epi_{}epo_{}rm_{}lr_nn_{}z_{}h_{}_'.format(args.num_ensembles, args.epochs_per_iter,
                                                             args.replay_memory_size,
                                                             args.lr_nn, args.z_dim, args.h_dim, args.scaling)
run_id = 'CARTPOLE_fixSTD:{}_{}ep_{}kl_{}gamma_pick{}_{}ctx_loo{}_no_z'.format(args.fixed_sigma, args.num_ensembles,
                                                                               args.max_kl,
                                                                               args.gamma, args.pick, args.num_context,
                                                                               args.loo) + dkl_spec

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

"""create policy, value function and agent TRPO"""
policy_net = Policy(state_dim, action_dim, log_std=args.log_std)
value_net = Value(state_dim)
policy_net.to(device)
value_net.to(device)

agent_trpo = Agent(env, policy_net, device, running_state=running_state, render=args.render, num_threads=1)

def update_params_trpo(batch):
    # (3)
    states = torch.from_numpy(np.stack(batch.state)).to(args.dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(args.dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(args.dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(args.dtype).to(device)
    with torch.no_grad():
        values = value_net(states)  # estimate value function of each state with NN

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """perform TRPO update"""
    trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl_trpo, args.damping, args.l2_reg)


'''create mean interpolator policy model'''

model = MeanInterpolator(state_dim, args.h_dim, args.z_dim, scaling=args.scaling).to(device).double()

optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters(), 'lr': args.lr_nn},
    {'params': model.interpolator.parameters(), 'lr': args.lr_nn}])

# train
model_trainer = MITrainer(device, model, optimizer, args, print_freq=30)

"""create replay memory"""
replay_memory = ReplayMemoryDataset(args.replay_memory_size, use_mean=True)

"""create agent"""
if args.pick:
    agent_mi = AgentPicker(env, model, args.device_np, args.num_context, running_state=running_state, render=args.render,
                        pick_dist=None, fixed_sigma=args.fixed_sigma)
else:

    agent_mi = Agent_all_ctxt(env, model, args.device_np, custom_reward=None, attention=False, mean_action=False,
                  render=args.render,
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
        all_states, all_means, all_stdv, all_actions = merge_padded_lists(
            [episode['states'] for episode in complete_dataset],
            [episode['means'] for episode in complete_dataset],
            [episode['stddevs'] for episode in complete_dataset],
            [episode['actions'] for episode in complete_dataset],
            max_lens=[episode['real_len'] for episode in complete_dataset])
        all_advantages = torch.cat(estimated_adv, dim=0).view(-1)
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
    if args.loo:
        model_trainer.train_rl_loo(data_loader, args.epochs_per_iter, early_stopping=None)
    else:
        model_trainer.train_rl(data_loader, args.epochs_per_iter, early_stopping=None)




def estimate_v_a(complete_dataset, disc_rew):
    ep_rewards = [tensor(rews) for rews in disc_rew]
    ep_states = [ep['states'] for ep in complete_dataset]
    real_lens = [ep['real_len'] for ep in complete_dataset]
    estimated_advantages = []
    for i in range(len(ep_states)):
        context_list = []
        j = 0
        for states, rewards, real_len in zip(ep_states, ep_rewards, real_lens):
            if j != i:
                context_list.append([states.unsqueeze(0), rewards.view(1, -1, 1), real_len])
            else:
                s_target = states[:real_len, :].unsqueeze(0)
                r_target = rewards.view(1, -1, 1)
            j += 1
        s_context, r_context = merge_context(context_list)
        with torch.no_grad():
            values = model(s_context, r_context, s_target)
        advantages = r_target - values
        estimated_advantages.append(advantages.squeeze(0))
    return estimated_advantages

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
                              sigma * torch.ones([1, max_episode_len, action_dim])).sample()
        initial_episodes.append([states, actions_init, max_episode_len])
    return initial_episodes


def create_directories(directory_path):
    os.mkdir(directory_path)
    os.mkdir(directory_path + '/Mean improvement/')
    os.mkdir(directory_path + '/z/')

avg_rewards_trpo = [0]
tot_steps_trpo = [0]
avg_rewards_mi = [0]
tot_steps_mi = [0]
improved_context_list = sample_initial_context_normal(args.num_ensembles)


def main_loop(improved_context_list):
    colors = []
    num_episodes = args.num_ensembles
    for i in range(num_episodes):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    for i_iter in range(args.max_iter_num):
        if tot_steps_trpo[-1] - tot_steps_mi[-1] < 1000:
            batch_trpo, log_trpo, memory_trpo = agent_trpo.collect_samples(args.min_batch_size)  # batch of batch_size transitions from multiple
            update_params_trpo(batch_trpo)  # generate multiple trajectories that reach the minimum batch_size
            tot_steps_trpo.append(tot_steps_trpo[-1] + log_trpo['num_steps'])
            avg_rewards_trpo.append(log_trpo['avg_reward'])

        batch_mi, log_mi = agent_mi.collect_episodes(improved_context_list, render=(i_iter % 10 == 0))
        disc_rew = discounted_rewards(batch_mi.memory, args.gamma)
        complete_dataset = BaseDataset(batch_mi.memory, disc_rew, args.device_np, args.dtype, max_len=max_episode_len)
        advantages = estimate_v_a(complete_dataset, disc_rew)

        t0 = time.time()
        improved_context_list = improvement_step_all(complete_dataset, advantages)
        t1 = time.time()

        # create training set
        tn0 = time.time()
        replay_memory.add(complete_dataset)
        train_np(replay_memory)
        tn1 = time.time()

        tv0 = time.time()
        if False and i_iter % args.plot_every == 0:
            # plot_initial_context(improved_context_list, colors, env, args, i_iter)
            # plot_training_set(i_iter, replay_memory, env, args)
            # plot_policy(model, improved_context_list, replay_memory, i_iter, log['avg_reward'], env, args, colors)
            plot_improvements(complete_dataset, disc_rew, env, i_iter, args, colors)
        tv1 = time.time()
        tot_steps_mi.append(tot_steps_mi[-1] + log_mi['num_steps'])
        avg_rewards_mi.append(log_mi['avg_reward'])
        if i_iter % args.log_interval == 0:
            print('{}\n R_min_trpo {:.2f} \tR_max_trpo {:.2f} \tR_avg_trpo {:.2f}\nR_min_mi {:.2f} \tR_max_mi {:.2f} \tR_avg_mi {:.2f} '.format(
                i_iter, log_trpo['min_reward'], log_trpo['max_reward'], log_trpo['avg_reward'],
                 log_mi['min_reward'], log_mi['max_reward'], log_mi['avg_reward']))

        if i_iter % args.plot_every == 0:
            plot_rewards_history(trpo=[tot_steps_trpo, avg_rewards_trpo], mi=[tot_steps_mi, avg_rewards_mi], args=args)
    plot_rewards_history(trpo=[tot_steps_trpo, avg_rewards_trpo], mi=[tot_steps_mi, avg_rewards_mi], args=args)

    """clean up gpu memory"""
    torch.cuda.empty_cache()

def plot_rewards_history(trpo=None, mi=None, args=args):
    fig_rew, ax_rew = plt.subplots(1, 1)
    colors = ['r', 'b']
    labels = ['trpo', 'mean interpolation']
    for i, log in enumerate([trpo, mi]):
        tot_steps, avg_rewards = log
        ax_rew.plot(tot_steps, avg_rewards, c=colors[i], label=labels[i])
    ax_rew.set_xlabel('number of steps')
    ax_rew.set_ylabel('average reward')
    ax_rew.set_title('Average Reward History')
    plt.legend()
    plt.grid()
    fig_rew.savefig(args.directory_path + run_id.replace('.', ','))
    plt.close(fig_rew)

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


def plot_policy(policy_np, all_context_xy, rm, iter_pred, avg_rew, env, args, colors):
    size = 8
    fig = plt.figure(figsize=(16, 8))
    x, X1, X2, X3, X4, xs = create_plot_4d_grid(env, args, size=size)
    mu_list = []
    xp1, xp2 = np.meshgrid(xs[0], xs[2])
    xp3, xp4 = np.meshgrid(xs[1], xs[3])
    middle_vel = len(X2) // 2
    ax1c = fig.add_subplot(2, 2, 2, projection='3d')
    ax2c = fig.add_subplot(2, 2, 4, projection='3d')
    x_context, y_context = merge_context(all_context_xy)
    ax1c.scatter(x_context[0, :, [0]].cpu(), x_context[0, :, [2]].cpu(), y_context.view(-1, 1).cpu(), cmap='viridis',
                 vmin=-1.,
                 vmax=1.)
    ax2c.scatter(x_context[0, :, [1]].cpu(), x_context[0, :, [3]].cpu(), y_context.view(-1, 1).cpu(), cmap='viridis',
                 vmin=-1.,
                 vmax=1.)
    with torch.no_grad():
        p_y_pred = model(x_context, y_context, x[0:1])
        mu = p_y_pred.view(X1.shape).cpu().numpy()
        mu_list.append(mu)

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 3, projection='3d')

    for z_mean in mu_list:
        ax1.plot_surface(xp1, xp2, z_mean[:, middle_vel, :, middle_vel], cmap='viridis', vmin=-1., vmax=1.)
        ax2.plot_surface(xp3, xp4, z_mean[middle_vel, :, middle_vel, :], cmap='viridis', vmin=-1., vmax=1.)

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

    fig.savefig(args.directory_path + str(iter_pred), dpi=250)
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
    set_bounds([ax, ax_rew], [0, 2])
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
            ax.scatter(states[:, 0].numpy(), states[:, 2].numpy(), means[:, 0].numpy(), c='k', label='sampled',
                       alpha=0.3)
            ax.scatter(states[:, 0].numpy(), states[:, 2].numpy(), new_means[:, 0].numpy(), c=new_means[:, 0].numpy(),
                       marker='+', label='improved', alpha=0.6)
            leg = ax.legend(loc="upper right")
        else:
            ax.scatter(states[:, 0].numpy(), states[:, 2].numpy(), means[:, 0].numpy(), c='k', alpha=0.3)
            ax.scatter(states[:, 0].numpy(), states[:, 2].numpy(), new_means[:, 0].numpy(), c=new_means[:, 0].numpy(),
                       marker='+', alpha=0.6)

        a = ax_rew.scatter(states[:, 0].numpy(), states[:, 2].numpy(), actions[:, 0].numpy(), c=est_rew[:],
                           cmap='viridis', alpha=0.5)

    cb = fig.colorbar(a)
    cb.set_label('Discounted rewards')
    ax_rew.set_title('Discounted rewards')
    fig.savefig(args.directory_path + '/Mean improvement/' + name, dpi=250)
    plt.close(fig)


#create_directories(args.directory_path)
main_loop(improved_context_list)
