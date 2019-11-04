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
from models.mlp_policy_disc import DiscretePolicy
from core.trpo import trpo_step
from core.common import estimate_advantages
from core.agent import Agent
from neural_process import NeuralProcess
from training_module_RL import NeuralProcessTrainerRL

torch.set_default_tensor_type(torch.DoubleTensor)

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="MountainCarContinuous-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--use-neural-process', default=False,
                    help='whether to use NP or TRPO as policy')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
parser.add_argument('--num-threads', type=int, default=5, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=13, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=4000, metavar='N',
                    help='minimal batch size per TRPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=501, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
args = parser.parse_args()


dtype = torch.float64
torch.set_default_dtype(dtype)
device_np = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
device_rl = torch.device("cpu")

"""environment"""
use_running_state = False
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
policy_np = NeuralProcess(state_dim, action_dim, 32, 12, 32).to(device_np)
optimizer = torch.optim.Adam(policy_np.parameters(), lr=3e-4)
np_trainer = NeuralProcessTrainerRL(device_np, policy_np, optimizer,
                                    num_context_range=(400, 500),
                                    num_extra_target_range=(400, 500),
                                    print_freq=2000)
"""create replay memory"""
replay_memory_size = 10
replay_memory = ReplayMemoryDataset(replay_memory_size)

"""define actor and critic"""
if args.use_neural_process:
    policy_net = policy_np
    print('using np as policy')
else:
    policy_net = Policy(state_dim, action_dim, log_std=args.log_std)
    value_net = Value(state_dim)
    policy_net.to(device_rl)
    value_net.to(device_rl)

"""create agent"""
agent = Agent(env, policy_net, device_rl, running_state=running_state, render=args.render, num_threads=args.num_threads)



def improvement_step(actions):
    print('improving action')
    grad = np.random.randn()
    step = 0.1
    return actions + step*grad*actions


def update_params_trpo(batch):
    # (3)
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device_rl)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device_rl)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device_rl)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device_rl)
    with torch.no_grad():
        values = value_net(states)  # estimate value function of each state with NN

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device_rl)

    """perform TRPO update"""
    trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg)


run_id = '(train 4k)'
directory_path = '/home/francesco/PycharmProjects/MasterThesis/RL memories/TRPO policies&samples MCC' + run_id

def set_labels(ax):
    ax.set_xlabel('Position', fontsize=14)
    ax.set_ylabel('Velocity', fontsize=14)
    ax.set_zlabel('Acceleration', fontsize=14)

def set_limits(ax, env):
    bounds_high = env.observation_space.high
    bounds_low = env.observation_space.low
    ax.set_xlim(bounds_low[0], bounds_high[0])
    ax.set_ylim(bounds_low[1], bounds_high[1])
    ax.set_zlim(env.action_space.low, env.action_space.high)

num_test_context = 999
def plot_NP_policy(context_xy, iter_pred, num_samples=1):
    from mpl_toolkits.mplot3d import Axes3D
    # Axes3D import has side effects, it enables using projection='3d' in add_subplot
    import matplotlib.pyplot as plt
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

    name = 'NP trained on '+str(replay_memory_size)+' trajectories predicts: {}'.format(iter_pred)
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
    ax_context.set_title('Context points from one TRPO trajectory', pad=20, fontsize=16)
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

def plot_policy(policy_net, info):
    from mpl_toolkits.mplot3d import Axes3D
    # Axes3D import has side effects, it enables using projection='3d' in add_subplot
    import matplotlib.pyplot as plt
    fig = plt.figure()
    bounds_high = env.observation_space.high
    bounds_low = env.observation_space.low
    if not use_running_state:
        x1 = np.linspace(bounds_low[0], bounds_high[0], 100)
        x2 = np.linspace(bounds_low[1], bounds_high[1], 100)
    else:
        x1 = np.linspace(-2, 2, 100)
        x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)
    zs = []
    for _x1 in x1:
        for _x2 in x2:
            state = tensor([_x1, _x2], device=device_rl).unsqueeze(0)
            zs.append(policy_net(state)[0][0].item())
    Z = (np.array(zs)).reshape(X1.shape).transpose()

    ax = plt.axes(projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis',  vmin=-1., vmax=1.)

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Acceleration')
    ax.set_zlim(env.action_space.low, env.action_space.high)
    name = 'TRPO on {} iter: {} avg_rew: {}'.format(args.env_name, info[0], int(info[1]))
    ax.set_title(name, pad=20)
    fig.savefig(directory_path+'/TRPO policies/'+name)
    plt.close(fig)
    # plt.show()

def create_directories(directory_path):
    os.mkdir(directory_path)
    os.mkdir(directory_path+'/TRPO policies/')
    os.mkdir(directory_path+'/NP estimate/')

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

np_batch_size = 4
epochs_per_iter = 10
def train_np(datasets):
    policy_np.training = True
    data_loader = DataLoader(datasets, batch_size=np_batch_size, shuffle=True)
    np_trainer.train(data_loader, epochs_per_iter)
    policy_np.training = False


def main_loop():
    for i_iter in range(args.max_iter_num):
        # (1)
        # generate multiple trajectories that reach the minimum batch_size
        # introduce param context=None when np is policy, these will be the context points used to predict
        batch, log, memory = agent.collect_samples(args.min_batch_size)  # batch of batch_size transitions from multiple
        print(log['num_steps'], log['num_episodes'])                     # episodes (separated by mask=0). Stored in Memory
        t0 = time.time()
        if not args.use_neural_process:
            update_params_trpo(batch)  # estimate advantages from samples and update policy by TRPO step
        else:
            improved_actions = improvement_step(batch)
            context_for_agent_collect_samples = improved_actions
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1 - t0, log['min_reward'], log['max_reward'], log['avg_reward']))
            plot_policy(policy_net, (i_iter, log['avg_reward']))

        with torch.no_grad():
            dataset = MemoryDataset(memory.memory, device_np, dtype, max_len=999)
            if i_iter > 0:
                # predict with NP before training on last episode
                x, y, num_steps = dataset.data[0]
                x_context, y_context = sample_context(x.unsqueeze(0), y.unsqueeze(0), min(num_steps, num_test_context))
                plot_NP_policy([x_context, y_context], i_iter, num_samples=1)

            replay_memory.add(dataset)
        print('replay memory size:', len(replay_memory))
        train_np(replay_memory)


    """clean up gpu memory"""
    torch.cuda.empty_cache()

create_directories(directory_path)
main_loop()
