import argparse
import gym
import os
import sys
import pickle
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from random import randint
import gpytorch
from utils_rl import *

from training_module_RL import NeuralProcessTrainerRL
from neural_process import NeuralProcess
from torch.utils.data import Dataset, DataLoader
from multihead_attention_np import AttentiveNeuralProcess
from utils.utils import context_target_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

parent_dir = '/home/francesco/PycharmProjects/MasterThesis/RL memories/TRPO policies&samples MCC(9)/'
folder_name = '5 datasets all context plotted/multihead/'
memory_dir = parent_dir + 'episodes/'

num_memories = 501
plot_one_traj = False  # plots all context points used (multiple trajectories)
test_contexts = []
num_test_context = 999
num_training_datasets = 5

env_name = "MountainCarContinuous-v0"
env = gym.make(env_name)
x_dim = env.observation_space.shape[0]
y_dim = env.action_space.shape[0]

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default=env_name, metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tot-iter', default=num_memories,  metavar='G',
                    help='number of stored TRPO iterations')
# NP parameters
parser.add_argument('--epochs-per-iter', default=20, metavar='G',
                    help='epochs of training for every memory')
parser.add_argument('--x-dim', default=x_dim, metavar='G',
                    help='observation space')
parser.add_argument('--y-dim', default=y_dim, metavar='G',
                    help='action space')
parser.add_argument('--z-dim', default=128, metavar='G',
                    help='latent space')
parser.add_argument('--a-dim', default=128, metavar='G',
                    help='attention space')
parser.add_argument('--h-dim', default=256, metavar='G',
                    help='hidden layer dim')
parser.add_argument('--r-dim', default=256, metavar='G',
                    help='representation space')
parser.add_argument('--batch-size', default=4, metavar='G',
                    help='number of trajectories in a batch')
parser.add_argument('--att-type', default='dot_product', metavar='G',
                    help='attention type')
args = parser.parse_args()

use_attention = False

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

def plot_NP_policy(neural_process, context_xy, first, last, num_samples=1):
    from mpl_toolkits.mplot3d import Axes3D
    # Axes3D import has side effects, it enables using projection='3d' in add_subplot
    import matplotlib.pyplot as plt
    bounds_high = env.observation_space.high
    bounds_low = env.observation_space.low
    x1 = np.linspace(bounds_low[0], bounds_high[0], 100)
    x2 = np.linspace(bounds_low[1], bounds_high[1], 100)
    X1, X2 = np.meshgrid(x1, x2)

    grid = torch.zeros(100, 2)
    for i in range(2):
        grid_diff = float(bounds_high[i] - bounds_low[i]) / (100 - 2)
        grid[:, i] = torch.linspace(bounds_low[i] - grid_diff, bounds_high[i] + grid_diff, 100)

    x = gpytorch.utils.grid.create_data_from_grid(grid)

    # Plot a realization
    Z_distr = neuralprocess(context_xy[0], context_xy[1], x.unsqueeze(0))  # B x num_points x z_dim  (B=1)
    Z_mean = Z_distr.mean.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
    Z_stddev = Z_distr.stddev.detach()[0].reshape(X1.shape) # x1_dim x x2_dim

    name = 'NP trained on {}-{} predicts: {}'.format(first, last, last+1)
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
    if plot_one_traj:
        ax_context.set_title('Context points from one TRPO trajectory', pad=20, fontsize=16)
        z = context_xy[1][0,:,0].detach().cpu().numpy()
        xs_context = context_xy[0][0,:,0].detach().cpu().numpy()
        ys_context = context_xy[0][0,:,1].detach().cpu().numpy()
        ax_context.scatter(xs_context, ys_context, z, s=8, c=z, cmap='viridis',  vmin=-1., vmax=1.)
    else:
        ax_context.set_title('Context points from all trained trajectories', pad=20, fontsize=16)
        for i in range(len(next_dataset)):
            x, y, num_steps = next_dataset.data[i]
            x_context, z_context = sample_context(x.unsqueeze(0), y.unsqueeze(0), min(num_steps, num_test_context))
            z = z_context[0, :, 0].detach().cpu().numpy()
            xs_context = x_context[0, :, 0].detach().cpu().numpy()
            ys_context = x_context[0, :, 1].detach().cpu().numpy()
            ax_context.scatter(xs_context, ys_context, z, s=8, c=z, cmap='viridis', vmin=-1., vmax=1.)

    ax_samples = fig.add_subplot(224, projection='3d')
    ax_samples.set_title(str(num_samples) + ' samples from policy', pad=20, fontsize=16)
    set_limits(ax_samples, env)
    set_labels(ax_samples)
    for sample in range(num_samples):
        Z_sample = Z_distr.sample().detach()[0].reshape(X1.shape)
        ax_samples.plot_surface(X1, X2, Z_sample.cpu().numpy(), cmap='viridis', vmin=-1., vmax=1., alpha=0.2)

    # plt.show()
    fig.savefig(parent_dir+folder_name+name, dpi=250)
    plt.close(fig)


def sample_context(x, y, num_context=100):
    num_points = x.shape[1]
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_context,
                                 replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    return x_context, y_context

if use_attention:
    neuralprocess = AttentiveNeuralProcess(args.x_dim, args.y_dim, args.r_dim, args.z_dim,
                                            args.h_dim, args.a_dim, use_self_att=True).to(device)
else:
    neuralprocess = NeuralProcess(args.x_dim, args.y_dim, args.r_dim, args.z_dim, args.h_dim).to(device)

optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=3e-4)
np_trainer = NeuralProcessTrainerRL(device, neuralprocess, optimizer,
                                    num_context_range=(400, 500),
                                    num_extra_target_range= (400, 500),
                                    print_freq=2)


def get_dataset(i_iter):
    file_name = memory_dir + str(i_iter) + '^iter_' + env_name

    with open(file_name, 'rb') as file_m:
        memory_iter = pickle.load(file_m)  # memory_iter.memory to access list of transitions

    dataset = MemoryDataset(memory_iter.memory, max_len=999)
    return dataset


for i_iter in range(args.tot_iter-1):
    print('start training on ', i_iter)
    neuralprocess.training = True
    dataset = get_dataset(i_iter)
    start_dataset = max(0, i_iter-num_training_datasets)
    for d in range(start_dataset, i_iter):
        print('added ',d)
        add_dataset = get_dataset(d)
        dataset.data += add_dataset.data

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    np_trainer.train(data_loader, args.epochs_per_iter)

    """plot results"""
    next_dataset = get_dataset(i_iter+1)
    x, y, num_steps = next_dataset.data[0]
    x_context, y_context = sample_context(x.unsqueeze(0), y.unsqueeze(0), min(num_steps, num_test_context))
    neuralprocess.training = False

    plot_NP_policy(neuralprocess, [x_context, y_context], start_dataset, i_iter, num_samples=1)

    print('predicted policy ', i_iter+1)
