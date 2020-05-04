import argparse
import gym
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gpytorch
from utils_rl import *
from core.common import discounted_rewards
from previous_methods.previous_agents.agent_np import Agent
from neural_process import NeuralProcess
from training_module_RL import NeuralProcessTrainerRL
import matplotlib.pyplot as plt
from torch.distributions import Normal

# Axes3D import has side effects, it enables using projection='3d' in add_subplot
torch.set_default_tensor_type(torch.DoubleTensor)

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="MountainCarContinuous-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--use-attention', default=False,
                    help='whether to use NP or TRPO as policy')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--epochs-per-iter', type=int, default=20, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--replay-memory-size', type=int, default=15, metavar='G',
                    help='size of training set in episodes')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=7, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=15*999, metavar='N',
                    help='minimal batch size per TRPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=501, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
args = parser.parse_args()

replay_memory_size = 100
improve_mean = False   # whether to use the improved mean or actions sampled from them as context points
sample_improved_action = True  # sample actions from improved mean or improve old actions
use_running_state = False
coherent = False

frac_replace_actions = 1  # replace mean with actions in training set

# NP params
r_dim = 256
z_dim = 128
h_dim = 256
np_batch_size = 8

np_spec = '{}z_{}rm_{}e_imprM:{}_sampled_a:{}_RS:{}_replace{}_coh:{}'.format(z_dim, replay_memory_size, args.epochs_per_iter, improve_mean,
                                                                      sample_improved_action, use_running_state, frac_replace_actions, coherent)
run_id = np_spec + '{}b_{}kl_{}gamma_'.format(args.min_batch_size, args.max_kl, args.gamma)
directory_path = '/home/francesco/PycharmProjects/MasterThesis/NP learning results/' + run_id

dtype = torch.float64
torch.set_default_dtype(dtype)
device_np = torch.device('cpu')  #torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
if use_running_state:
    running_state = ZFilter((state_dim,), clip=5)  # running list of states that allows to access precise mean and std
else:
    running_state = None
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

'''create neural process'''
policy_np = NeuralProcess(state_dim, action_dim, r_dim, z_dim, h_dim).to(device_np)
optimizer = torch.optim.Adam(policy_np.parameters(), lr=3e-4)
np_trainer = NeuralProcessTrainerRL(device_np, policy_np, optimizer,
                                    num_context_range=(400, 500),
                                    num_extra_target_range=(400, 500),
                                    print_freq=100)
"""create replay memory"""
replay_memory = ReplayMemoryDataset(replay_memory_size)


"""create agent"""
agent = Agent(env, policy_np, device_np, running_state=running_state, render=args.render, num_threads=args.num_threads, mean_action=coherent)

def estimate_eta(episode):
    d = episode[0].action.shape[0]
    if d > 1:
        raise NotImplementedError('compute eta not implemented for action space of dim>1')
    else:
        iter_sum = 0
        eps = tensor(args.max_kl).to(dtype)
        T = tensor(len(episode)).to(dtype)
        for t in episode:
            iter_sum += ((t.disc_rew ** 2) * (t.action - t.mean) ** 2) / t.stddev ** 4
        denominator = torch.from_numpy(iter_sum).to(dtype)
        return torch.sqrt((T*eps)/denominator)


def improvement_step(memory):
    print('improving actions')
    first = True
    for episode in memory:
        eta = estimate_eta(episode)
        for transition in episode:
            state = torch.from_numpy(transition.state).to(dtype)
            action = torch.from_numpy(transition.action).to(dtype)
            mean = torch.from_numpy(transition.mean).to(dtype)
            stddev = torch.from_numpy(transition.stddev).to(dtype)
            discounted_reward = tensor(transition.disc_rew).to(dtype).unsqueeze(0)

            new_mean = mean + eta*discounted_reward*((action-mean)/stddev)
            if sample_improved_action:
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



def set_labels(ax):
    ax.set_xlabel('Position', fontsize=14)
    ax.set_ylabel('Velocity', fontsize=14)
    ax.set_zlabel('Acceleration', fontsize=14)

def set_limits(ax, env):
    if use_running_state:
        ax.set_zlim(env.action_space.low, env.action_space.high)
        return
    bounds_high = env.observation_space.high
    bounds_low = env.observation_space.low
    ax.set_xlim(bounds_low[0], bounds_high[0])
    ax.set_ylim(bounds_low[1], bounds_high[1])
    ax.set_zlim(env.action_space.low, env.action_space.high)

num_test_context = 999
def plot_NP_policy(context_xy, iter_pred, avg_rew, num_samples=1):
    policy_np.training = False
    bounds_high = env.observation_space.high
    bounds_low = env.observation_space.low
    if not use_running_state:
        x1 = np.linspace(bounds_low[0], bounds_high[0], 100)
        x2 = np.linspace(bounds_low[1], bounds_high[1], 100)
    else:
        x1 = np.linspace(-2, 2, 100)
        x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)

    grid = torch.zeros(100, 2)
    for i in range(2):
        grid_diff = float(bounds_high[i] - bounds_low[i]) / (100 - 2)
        grid[:, i] = torch.linspace(bounds_low[i] - grid_diff, bounds_high[i] + grid_diff, 100)

    x = gpytorch.utils.grid.create_data_from_grid(grid)
    x = x.unsqueeze(0).to(dtype).to(device_np)
    # Plot a realization
    Z_distr = policy_np(context_xy[0], context_xy[1], x)  # B x num_points x z_dim  (B=1)
    Z_mean = Z_distr.mean.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
    Z_stddev = Z_distr.stddev.detach()[0].reshape(X1.shape) # x1_dim x x2_dim

    name = 'NP policy for iteration {}, avg rew {}'.format(iter_pred, int(avg_rew))
    fig = plt.figure(figsize=(16,14)) #figsize=plt.figaspect(1.5)
    fig.suptitle(name, fontsize=20)
    fig.tight_layout()
    # fig.subplots_adjust(top=0.5, wspace=0.3, bottom=0.2)
    ax_mean = fig.add_subplot(221, projection='3d')
    ax_mean.plot_surface(X1, X2, Z_mean.cpu().numpy(), cmap='viridis',  vmin=-1., vmax=1.)
    set_labels(ax_mean)
    set_limits(ax_mean, env)

    ax_mean.set_title('Mean of the NP policy', pad=20, fontsize=16)

    ax_stdv = fig.add_subplot(222, projection='3d')
    set_limits(ax_stdv, env)
    set_labels(ax_stdv)
    ax_stdv.set_title('Standard deviation of the NP policy', pad=20, fontsize=14)
    stddev_low = Z_mean - Z_stddev
    stddev_high = Z_mean + Z_stddev

    i = 0
    for y_slice in x2:
        ax_stdv.add_collection3d(
            plt.fill_between(x1, stddev_low[i, :].cpu(), stddev_high[i, :].cpu(), color='lightseagreen', alpha=0.2),
            zs=y_slice, zdir='y')
        i += 1

    ax_context = fig.add_subplot(223, projection='3d')
    set_labels(ax_context)
    set_limits(ax_context, env)
    ax_context.set_title('Context points improved', pad=20, fontsize=16)
    z = context_xy[1][0,:,0].detach().cpu().numpy()
    xs_context = context_xy[0][0,:,0].detach().cpu().numpy()
    ys_context = context_xy[0][0,:,1].detach().cpu().numpy()
    ax_context.scatter(xs_context, ys_context, z, s=8, c=z, cmap='viridis',  vmin=-1., vmax=1.)


    ax_samples = fig.add_subplot(224, projection='3d')
    ax_samples.set_title(str(num_samples) + ' samples from policy', pad=20, fontsize=16)
    set_limits(ax_samples, env)
    set_labels(ax_samples)
    for sample in range(num_samples):
        Z_sample = Z_distr.sample().detach()[0].reshape(X1.shape)
        ax_samples.plot_surface(X1, X2, Z_sample.cpu().numpy(), cmap='viridis', vmin=-1., vmax=1., alpha=0.2)

    # plt.show()
    fig.savefig(directory_path+'/NP estimate/'+name, dpi=250)
    plt.close(fig)


def plot_training_set(i_iter):
    name = 'Training trajectories ' + str(i_iter)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_title(name)
    set_limits(ax, env)
    set_labels(ax)
    for trajectory in replay_memory:
        z = trajectory[1][:,0].detach().cpu().numpy()
        xs_context = trajectory[0][:,0].detach().cpu().numpy()
        ys_context = trajectory[0][:,1].detach().cpu().numpy()
        ax.scatter(xs_context, ys_context, z, c=z, cmap='viridis', alpha=0.1)
    fig.savefig(directory_path+'/Training/'+name)
    plt.close(fig)

def plot_improvements(batch, improved_context, i_iter):

    episode = batch[0]
    states = []
    means = []
    actions = []
    disc_rew  = []
    num_c = len(episode)
    for transition in episode:
        states.append(transition.state)
        disc_rew.append(transition.disc_rew)
        actions.append(transition.action)
        means.append(transition.mean)
    previous = means
    name = 'Improvement iter ' + str(i_iter)
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(name, fontsize=20)
    ax = fig.add_subplot(121, projection='3d')
    name = 'Context improvement iter ' + str(i_iter)
    ax.set_title(name)
    set_limits(ax, env)
    set_labels(ax)
    z = improved_context[1][0,:num_c,0].detach().cpu().numpy()
    xs_context = improved_context[0][0,:num_c,0].detach().cpu().numpy()
    ys_context = improved_context[0][0,:num_c,1].detach().cpu().numpy()
    state_1 = [state[0] for state in states]
    state_2 = [state[1] for state in states]
    ax.scatter(state_1, state_2, previous, c='r', label='sampled',  alpha=0.2)
    ax.scatter(xs_context, ys_context, z, c='y', label='improved', alpha=0.3)
    leg = ax.legend(loc="upper right")
    ax_rew = fig.add_subplot(122, projection='3d')
    set_labels(ax_rew)
    set_limits(ax_rew, env)
    a = ax_rew.scatter(state_1, state_2, actions, c=disc_rew, cmap='viridis', alpha=0.5)
    #plt.colorbar()
    cb = fig.colorbar(a)
    cb.set_label('Discounted rewards')
    ax_rew.set_title('Discounted rewards')
    fig.savefig(directory_path+'/Mean improvment/'+name)
    plt.close(fig)


def create_directories(directory_path):
    os.mkdir(directory_path)
    os.mkdir(directory_path+'/NP estimate/')
    os.mkdir(directory_path + '/Mean improvement/')
    os.mkdir(directory_path + '/Training/')

def sample_context(x, y, num_context=100):

    x = x.to(dtype).to(device_np)
    y = y.to(dtype).to(device_np)
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
    data_loader = DataLoader(datasets, batch_size=np_batch_size, shuffle=True)
    np_trainer.train(data_loader, args.epochs_per_iter, early_stopping=100)



def sample_initial_context(num_context, dtype=None):
    print('sampling initial context')
    def fix(tensor):
        return tensor.to(dtype).unsqueeze(0)
    all_states = fix(torch.from_numpy(env.observation_space.sample()))
    all_actions = fix(torch.from_numpy(env.action_space.sample()))
    for i in range(num_context-1):
        state = fix(torch.from_numpy(env.observation_space.sample()))
        action = fix(torch.from_numpy(env.action_space.sample()/3))
        all_states = torch.cat((all_states, state), dim=0)
        all_actions = torch.cat((all_actions, action), dim=0)
    #fig = plt.figure()
    #ax = plt.axes(projection="3d")
    #set_limits(ax, env)
    #set_labels(ax)
    #ax.scatter(all_states[:,0].numpy(), all_states[:,1].numpy(), all_actions[:,0].numpy(), c=all_actions[:,0].numpy(), cmap='viridis')
    #plt.show()
    return [all_states.unsqueeze(0), all_actions.unsqueeze(0)]


def main_loop():
    improved_context = sample_initial_context(args.min_batch_size, dtype=dtype)
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

        memory.set_disc_rew(disc_rew)
        t0 = time.time()
        all_improved_context = improvement_step(batch)
        t1 = time.time()
        key = 'means' if improve_mean else 'actions'
        improved_context = [all_improved_context['states'], all_improved_context[key]]

        # plot improved context and actions' discounted rewards
        plot_improvements(batch, [all_improved_context['states'], all_improved_context['means']], i_iter)

        # create training set
        training_set = all_improved_context['means']
        num_action_in_training = int(frac_replace_actions * training_set.shape[1])
        print('replacing {} means with actions'.format(num_action_in_training))
        training_set[:, :num_action_in_training, :] = all_improved_context['actions'][:, :num_action_in_training, :]

        dataset = MemoryDatasetNP(batch, training_set, device_np, dtype, max_len=999)
        replay_memory.add(dataset)

        plot_training_set(i_iter)

        print('replay memory size:', len(replay_memory))
        train_np(replay_memory)

        plot_NP_policy(improved_context, i_iter, log['avg_reward'])

        avg_rewards.append(log['avg_reward'])
        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                  i_iter, log['sample_time'], t1 - t0, log['min_reward'], log['max_reward'], log['avg_reward']))


    fig_rew, ax_rew = plt.subplots(1, 1)
    ax_rew.plot(np.arange(len(avg_rewards)), avg_rewards)
    ax_rew.set_xlabel('iterations')
    ax_rew.set_ylabel('average reward')
    fig_rew.savefig(directory_path+'/average reward')
    plt.close(fig_rew)
    """clean up gpu memory"""
    torch.cuda.empty_cache()

create_directories(directory_path)
main_loop()
