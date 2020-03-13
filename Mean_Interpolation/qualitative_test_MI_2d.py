import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
import gpytorch
from utils import context_target_split
from plotting_functions_DKL import plot_posterior
from torch.utils.data import DataLoader
from dataset_generator import SineData, GPData2D
from MeanInterpolatorModel import MeanInterpolator, MITrainer
import os
from plotting_functions_DKL import  create_plot_grid

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')
print('device: ', device)

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="Pendulum-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--seed', type=int, default=7, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--x-dim', type=int, default=2, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--y-dim', type=int, default=1, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--z-dim', type=int, default=4, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--h-dim', type=int, default=256, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--epochs', type=int, default=500, metavar='G',
                    help='training epochs')
parser.add_argument('--scaling', default='uniform', metavar='N',
                    help='z scaling')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--num-tot-samples', type=int, default=50, metavar='N',
                    help='batch size for np training')
parser.add_argument('--num-context', type=int, default=1000, metavar='N',
                    help='num context points')
parser.add_argument('--num-target', type=int, default=2500, metavar='N',
                    help='num target points')
parser.add_argument('--grid-size', type=int, default=100, metavar='N',
                    help='dimension of plotting grid')
parser.add_argument('--extent', default=(-1.,1.,-1.,1.), metavar='N',
                    help='')
parser.add_argument('--x-range', default=[-3, 3], metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--directory-path', default='/home/francesco/PycharmProjects/MasterThesis/plots/MI/2D/',
                    help='path to plots folder')
args = parser.parse_args()

## seedings
np.random.seed(args.seed)
torch.manual_seed(args.seed)

########################
#####  Parameters  #####
########################

# dataset parameters
kernel = 'matern'  # possible kernels ['RBF', 'cosine', 'linear', 'LCM', 'polynomial', 'periodic']
mean = ['linear']  # possible means ['linear', 'constant']
learning_rate = 1e-4
l = '3e-4'
x_range = args.x_range
grid_bounds=[args.extent[0:2], args.extent[2:4]]
id = 'MI_{}fcts_{}e_{}b_{}lr_{}z_{}h_{}_{}ctxt_2layers_'.format(args.num_tot_samples, args.epochs, args.batch_size,
                                              l, args.z_dim, args.h_dim, args.scaling, args.num_context)

args.directory_path += id

# Create dataset
dataset = dataset = GPData2D('constant', kernel, num_samples=args.num_tot_samples, grid_bounds=grid_bounds, grid_size=args.grid_size)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


test_dataset = dataset = GPData2D('constant', kernel, num_samples=args.num_tot_samples, grid_bounds=grid_bounds, grid_size=args.grid_size)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

print('dataset created')
# create model
model = MeanInterpolator(args.x_dim, args.h_dim, args.z_dim, scaling=args.scaling).to(device)



optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters(), 'lr':learning_rate},
    {'params': model.interpolator.parameters(), 'lr':1e-2}])
try:
    os.mkdir(args.directory_path)
except FileExistsError:
    pass

model_trainer = MITrainer(device, model, optimizer, num_context=args.num_context, num_target=args.num_target, print_freq=10)
print('start training')
model_trainer.train(data_loader, args.epochs, early_stopping=10.)

# Visualize data samples
plt.figure(1)

f, axarr = plt.subplots(2,2)

axarr[0, 0].imshow(dataset.data[0][1].view(-1, args.grid_size).cpu().numpy(), extent=args.extent)
axarr[0, 1].imshow(dataset.data[1][1].view(-1, args.grid_size).cpu().numpy(), extent=args.extent)
axarr[1, 0].imshow(dataset.data[2][1].view(-1, args.grid_size).cpu().numpy(), extent=args.extent)
axarr[1, 1].imshow(dataset.data[3][1].view(-1, args.grid_size).cpu().numpy(), extent=args.extent)
name = 'Samples from 2D gp with kernel: '+kernel
plt.savefig(args.directory_path + kernel + '_data')
axarr[0,0].set_title(name, pad=20)
for ax in axarr.flat:
    ax.set(xlabel='x1', ylabel='x2')
    ax.label_outer()

plt.close(f)


if args.epochs > 1:
    plt.figure(2)
    plt.title('average loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('average loss')
    plt.plot(np.linspace(0, args.epochs-1 ,args.epochs), model_trainer.epoch_loss_history, c='b', alpha=0.5)
    plt.grid()
    plt.savefig(args.directory_path + '/_loss_history_'+id)
    plt.close()



def plot_posterior_2d(data_loader, model, id, args):
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
    ax_real.plot_surface(X1, X2, y.reshape(X1.shape).cpu().numpy(), cmap='viridis')
    ax_real.set_title('Real function')

    ax_context = fig.add_subplot(132, projection='3d')
    ax_context.scatter(x_context[0,:,0].detach().cpu().numpy(),
                       x_context[0, :, 1].detach().cpu().numpy(),
                       y_context[0,:,0].detach().cpu().numpy(),
                       c=y_context[0,:,0].detach().cpu().numpy(),
                       cmap='viridis', vmin=-1., vmax=1.,  s=8)

    ax_context.set_title('Context points')
    with torch.no_grad():
        mu = model(x_context, y_context, x[0:1])


    ax_mean = fig.add_subplot(133, projection='3d')
    # Extract mean of distribution
    ax_mean.plot_surface(X1, X2, mu.cpu().view(X1.shape).numpy(), cmap='viridis')
    ax_mean.set_title('Posterior estimate')
    plt.savefig(args.directory_path + '/posteriior' + id, dpi=350)
    #plt.show()
    plt.close(fig)
    return
plot_posterior_2d(data_loader, model, id, args)
