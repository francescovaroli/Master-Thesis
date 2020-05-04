import argparse
import gym
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from new_plotting_functions import *
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
parser.add_argument('--max-kl', type=float, default=1e-3, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--min-batch-size', type=int, default=8*999, metavar='N',
                    help='minimal batch size per TRPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=501, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')

parser.add_argument('--epochs-per-iter', type=int, default=21, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--replay-memory-size', type=int, default=30, metavar='G',
                    help='size of training set in episodes')
parser.add_argument('--z-dim', type=int, default=256, metavar='N',
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

policy_np = NeuralProcess(2, 1, args.r_dim, args.z_dim, args.h_dim, fixed_sigma=None).to(args.device_np)
optimizer = torch.optim.Adam(policy_np.parameters(), lr=3e-4)
np_trainer = NeuralProcessTrainerRL(args.device_np, policy_np, optimizer,
                                    num_context_range=(400, 500),
                                    num_extra_target_range=(400, 500),
                                    print_freq=10)
env = gym.make(args.env_name)
def sample_initial_context_normal(num_episodes):
    initial_episodes = []
    max_episode_len = 999
    means_variance_gain = 1/100
    z_init = Normal(0, 1/means_variance_gain)
    for e in range(num_episodes):
        states = torch.zeros([1, max_episode_len, 2])
        for i in range(max_episode_len):
            states[:, i, :] = torch.from_numpy(env.observation_space.sample())

        z_sample = z_init.sample([1, args.z_dim])
        z_sample = z_sample.unsqueeze(1).repeat(1, max_episode_len, 1)
        means_init, stds_init = policy_np.xz_to_y(states, z_sample)
        actions_init = Normal(means_init, stds_init).sample()

        initial_episodes.append([states, actions_init, max_episode_len])
    return initial_episodes

def plot_initial_context(context_points, colors, env, args):
    name = 'Contexts of all episodes at iter '
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.set_title(name)
    set_limits(ax, env, args)
    set_labels(ax)
    for e, episode in enumerate(context_points):
        real_len = episode[2]
        z = episode[1][:, :real_len, 0].detach().cpu().numpy()
        xs_context = episode[0][:, :real_len, 0].detach().cpu().numpy()
        ys_context = episode[0][:, :real_len, 1].detach().cpu().numpy()
        ax.scatter(xs_context, ys_context, z, c=colors[e], alpha=0.5)
    fig.savefig(args.directory_path+'/test_pic', dpi=250)
    plt.close(fig)

print(args.env_name, args.max_kl)

colors = []
num_episodes = 10
for i in range(num_episodes):
    colors.append('#%06X' % randint(0, 0xFFFFFF))
improved_context_list = sample_initial_context_normal(10)
plot_initial_context(improved_context_list, colors, env, args)

'''
def plot_NP_initital(args, num_samples=1):
    num_test_context = 999
    policy_np.training = False
    bounds_high = (1, 1)
    bounds_low = (-1, -1)

    if not args.use_running_state:
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
        x = x.unsqueeze(0).to(args.dtype).to(args.device_np)
        fig = plt.figure(figsize=(16, 6))  # figsize=plt.figaspect(1.5)
        name = 'NP '
        fig.suptitle(name, fontsize=20)
      # fig.subplots_adjust(top=0.5, wspace=0.3, bottom=0.2)
        ax_mean = fig.add_subplot(121, projection='3d')
        ax_mean.set_title('Mean of the NP policy', pad=20, fontsize=16)

        ax_stdv = fig.add_subplot(122, projection='3d')
        ax_stdv.set_xlim(bounds_low[0], bounds_high[0])
        ax_stdv.set_ylim(bounds_low[1], bounds_high[1])
        vmin = 1000
        vmax = -1000

    for i in range(10):
            z_sample = torch.randn((1, args.z_dim))
            z_sample = z_sample.unsqueeze(1).repeat(1, 10000, 1)
            # Plot a realization
            Z_mean, Z_std = policy_np.xz_to_y(x, z_sample)  # B x num_points x z_dim  (B=1)
            Z_mean = Z_mean.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
            Z_stddev = Z_std.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
            stddev_low = Z_mean - Z_stddev
            stddev_high = Z_mean + Z_stddev
            vmin = min(stddev_low.min(), vmin)
            vmax = max(stddev_high.max(), vmax)


            ax_mean.plot_surface(X1, X2, Z_mean.cpu().numpy(), cmap='viridis', vmin=-1., vmax=1., alpha=0.4)



            ax_stdv.set_title('Standard deviation of the NP policy', pad=20, fontsize=14)


            i = 0

    for y_slice in x2:
            ax_stdv.add_collection3d(
                    plt.fill_between(x1, stddev_low[i, :].cpu(), stddev_high[i, :].cpu(), color='lightseagreen',
                                      alpha=0.02),
                    zs = y_slice, zdir = 'y')
            i += 1
    ax_mean.set_zlim(vmin, vmax)
    ax_stdv.set_zlim(vmin * 1.2, vmax * 1.2)

    plt.show()
    plt.close(fig)

plot_NP_initital(args)'''
