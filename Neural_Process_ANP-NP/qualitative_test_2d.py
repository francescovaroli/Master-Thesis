import numpy as np
import matplotlib.pyplot as plt
import torch
from random import randint
from torch.utils.data import DataLoader
from training_module import NeuralProcessTrainer
from neural_process import NeuralProcess
from network import LatentModel
from multihead_attention_np import *
#from attentive_neural_process import AttentiveNeuralProcess
from dataset_generator import SineData, GPData2D
from utils import context_target_split
import gpytorch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

plots_path = '/home/francesco/PycharmProjects/MasterThesis/plots/NP&ANP/2D/'

# settings
data = 'gp'
kernel_dict = ['RBF', 'cosine', 'linear', 'LCM', 'polynomial']
kernel = 'matern'

use_self_att = True
use_attention = True
att_type = 'dot_product'  # attention_types = ['uniform','laplace','dot_product']

epochs = 100

x_dim = 2
y_dim = 1
r_dim = 2*128  # Dimension of representation of context points
z_dim = 128  # Dimension of sampled latent variable
h_dim = 2*128  # Dimension of hidden layers in encoder and decoder
a_dim = 128

batch_size = 4
num_context = (190, 250)
num_target = (200, 350)

grid_bounds=[(-1,1),(-1,1)]
grid_size = 100

extent = (-1,1,-1,1)
# create ID for saving plots
mdl = 'NP'
if use_attention:
    mdl = 'A'+mdl
id = mdl + '{}e_{}b_{}c{}t_{}x{}x{}'.format(epochs, batch_size, num_context, num_target,r_dim, z_dim, h_dim)


# Create dataset
if data == 'sine':
    dataset = SineData(amplitude_range=(-2., 2.),
                       shift_range=(-.5, .5),
                       num_samples=2000)
elif data == 'gp':

    dataset = GPData2D('constant', kernel, num_samples=20, grid_bounds=grid_bounds, grid_size=grid_size)

# Visualize data samples
plt.figure(1)

f, axarr = plt.subplots(2,2)

axarr[0, 0].imshow(dataset.data[0][1].view(-1, grid_size).cpu().numpy(), extent=extent)
axarr[0, 1].imshow(dataset.data[1][1].view(-1, grid_size).cpu().numpy(), extent=extent)
axarr[1, 0].imshow(dataset.data[2][1].view(-1, grid_size).cpu().numpy(), extent=extent)
axarr[1, 1].imshow(dataset.data[3][1].view(-1, grid_size).cpu().numpy(), extent=extent)
name = 'Samples from 2D '+data+' with kernel: '+kernel
plt.savefig(plots_path + kernel + '_data')
axarr[0,0].set_title(name, pad=20)
for ax in axarr.flat:
    ax.set(xlabel='x1', ylabel='x2')
    ax.label_outer()

plt.show()
plt.close(f)

# create and train np
if use_attention:
    neuralprocess = AttentiveNeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim, a_dim, use_self_att=use_self_att).to(device)
else:
    neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim).to(device)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=3e-4)
np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                  num_context_range=num_context,
                                  num_extra_target_range= num_target,
                                  print_freq=500)

neuralprocess.training = True
np_trainer.train(data_loader, epochs)

plt.figure(2)
plt.title('average loss over epochs')
plt.plot(np.linspace(0, epochs-1 ,epochs), np_trainer.epoch_loss_history, c='b', alpha=0.5)
plt.savefig(plots_path + kernel + '_loss_history')
plt.show()
plt.close()

grid = torch.zeros(grid_size, len(grid_bounds))
for i in range(len(grid_bounds)):
    grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
    grid[:, i] = torch.linspace(grid_bounds[i][0] - grid_diff, grid_bounds[i][1] + grid_diff, grid_size)

x = gpytorch.utils.grid.create_data_from_grid(grid)
x = x.unsqueeze(0)

if not use_attention:
    mu_list = []
    for i in range(4):
        z_sample = torch.randn((1, z_dim))
        z_sample = z_sample.unsqueeze(1).repeat(1, x.size()[1], 1)
        mu, _ = neuralprocess.xz_to_y(x, z_sample)
        mu_list.append(mu)

    f2, axarr2 = plt.subplots(2,2)

    axarr2[0, 0].imshow(mu_list[0].view(-1, grid_size).detach().cpu().numpy(), extent=extent)
    axarr2[0, 1].imshow(mu_list[1].view(-1, grid_size).detach().cpu().numpy(), extent=extent)
    axarr2[1, 0].imshow(mu_list[2].view(-1, grid_size).detach().cpu().numpy(), extent=extent)
    axarr2[1, 1].imshow(mu_list[3].view(-1, grid_size).detach().cpu().numpy(), extent=extent)
    f2.suptitle('Samples from trained prior')
    for ax in axarr2.flat:
        ax.set(xlabel='x1', ylabel='x2')
        ax.label_outer()
    plt.savefig(plots_path + kernel + 'prior'+id)
    plt.show()
    plt.close(f2)

# Extract a batch from data_loader
for batch in data_loader:
    break

# Use batch to create random set of context points
x, y = batch
num_context_t = randint(*num_context)
num_target_t = randint(*num_target)
x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1],
                                                  num_context_t,
                                                  num_target_t)

neuralprocess.training = False

f3, axarr3 = plt.subplots(2,2)
axarr3[0,1].scatter(x_context[0].detach().cpu().numpy()[:,0],
                    x_context[0].detach().cpu().numpy()[:,1],
                    cmap='viridis', c=y_context[0].detach().cpu().numpy()[:,0],  s=1)
axarr3[0,0].imshow(y[0].view(-1, grid_size).detach().cpu().numpy()[::-1], extent=extent)
axarr3[0,0].set_title('Real function')
axarr3[0,1].set_xlim(extent[0], extent[1])
axarr3[0,1].set_ylim(extent[2], extent[3])
axarr3[0,1].set_aspect('equal')
axarr3[0,1].set_title('Context points')
mu_list = []
for i in range(2):
    # Neural process returns distribution over y_target
    p_y_pred = neuralprocess(x_context, y_context, x[0].unsqueeze(0))

    # Extract mean of distribution
    mu_list.append(p_y_pred.loc.detach())

axarr3[1, 0].imshow(mu_list[0].view(-1, grid_size).detach().cpu().numpy()[::-1], extent=extent)
axarr3[1,0].set_title('Posterior estimate_1')

axarr3[1, 1].imshow(mu_list[1].view(-1, grid_size).detach().cpu().numpy()[::-1], extent=extent)
axarr3[1,1].set_title('Posterior estimate_2')
plt.savefig(plots_path + kernel + ' posteriior'+id)
plt.show()
plt.close(f3)

torch.cuda.empty_cache()


def plot_posterior():
    for batch in data_loader:
        break

    # Use batch to create random set of context points
    x, y = batch  # , real_len

    x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1],
                                                      args.num_context,
                                                      args.num_target)
    x, X1, X2, x1, x2 = create_plot_grid(args.extent, args, size=args.grid_size)

    fig = plt.figure(figsize=(20, 8))  # figsize=plt.figaspect(1.5)
    fig.suptitle(id, fontsize=20)
    #fig.tight_layout()

    ax_real = fig.add_subplot(131, projection='3d')
    ax_real.plot_surface(X1, X2, y.reshape(X1.shape).cpu().numpy(), cmap='viridis', vmin=-1., vmax=1.)
    ax_real.set_title('Real function')

    ax_context = fig.add_subplot(132, projection='3d')
    ax_context.scatter(x_context[0,:,0].detach().cpu().numpy(),
                       x_context[0, :, 1].detach().cpu().numpy(),
                       y_context[0,:,0].detach().cpu().numpy(),
                       c=y_context[0,:,0].detach().cpu().numpy(),
                       cmap='viridis', vmin=-1., vmax=1.,  s=8)

    ax_context.set_title('Context points')
    model.set_train_data(x_context.squeeze(0), y_context.squeeze(0).squeeze(-1), strict=False)
    model.training = False
    with torch.no_grad():
        p_y_pred = model(x[0:1])
    mu = p_y_pred.mean.reshape(X1.shape).cpu()
    mu[torch.isnan(mu)] = 0.
    mu = mu.numpy()
    sigma = p_y_pred.stddev.reshape(X1.shape).cpu()
    sigma[torch.isnan(sigma)] = 0.
    sigma = sigma.numpy()
    std_h = mu + sigma
    std_l = mu - sigma
    model.training = True
    max_mu = std_h.max()
    min_mu = std_l.min()
    ax_mean = fig.add_subplot(133, projection='3d')
    i = 0
    for y_slice in x2:
        ax_mean.add_collection3d(
            plt.fill_between(x1, std_l[i, :], std_h[i, :], color='lightseagreen',
                             alpha=0.1),
            zs=y_slice, zdir='y')
        i += 1
    # Extract mean of distribution
    ax_mean.plot_surface(X1, X2, mu, cmap='viridis', vmin=-1., vmax=1.)
    for ax in [ax_mean, ax_context, ax_real]:
        ax.set_zlim(min_mu, max_mu)
    ax_mean.set_title('Posterior estimate')
    plt.savefig(args.directory_path + '/posteriior' + id, dpi=250)
    #plt.show()
    plt.close(fig)
    return
