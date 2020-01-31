import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
import gpytorch
from utils import context_target_split
from plotting_functions_DKL import plot_posterior_2d
from torch.utils.data import DataLoader
from dataset_generator import GPData2D
from DKModel import GPRegressionModel, DKMTrainer
import os

if torch.cuda.is_available() and False:
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
parser.add_argument('--z-dim', type=int, default=2, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--h-dim', type=int, default=100, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--epochs', type=int, default=2, metavar='G',
                    help='training epochs')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--num-tot-samples', type=int, default=35, metavar='N',
                    help='batch size for np training')
parser.add_argument('--num-context', type=int, default=100, metavar='N',
                    help='num context points')
parser.add_argument('--num-target', type=int, default=9900, metavar='N',
                    help='num target points')
parser.add_argument('--grid-size', type=int, default=100, metavar='N',
                    help='dimension of plotting grid')
parser.add_argument('--extent', default=(-.5,.5,-1,1), metavar='N',
                    help='dimension of latent variable in np')

parser.add_argument('--directory-path', default='/home/francesco/PycharmProjects/MasterThesis/plots/DKL/2D/',
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
mean = 'linear'  # possible means ['linear', 'constant']
learning_rate = 3e-4
l = '3e-4'
x_range = (-3., 3.)
grid_bounds=[(-.5,.5),(-1,1)]


id = 'DKM_{}fcts_{}c_{}t_{}e_{}b_{}lr_{}z_{}h_noise'.format(args.num_tot_samples, args.num_context, args.num_target,
                                                      args.epochs, args.batch_size,l, args.z_dim, args.h_dim)

args.directory_path += id+'/'

# Create dataset
dataset = dataset = GPData2D('constant', kernel, num_samples=args.num_tot_samples, grid_bounds=grid_bounds, grid_size=args.grid_size)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


test_dataset = dataset = GPData2D('constant', kernel, num_samples=args.num_tot_samples, grid_bounds=grid_bounds, grid_size=args.grid_size)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

for data_init in data_loader:
    break
x_init, y_init = data_init
x_init, y_init, _, _ = context_target_split(x_init[0:1], y_init[0:1],
                                                                args.num_context,
                                                                args.num_target)
#x_init, y_init = dataset.data[0]
print('dataset created')
# create model
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = GPRegressionModel(x_init, y_init.squeeze(0).squeeze(-1), likelihood,
                          args.h_dim, args.z_dim, name_id=id).to(device)


optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()}], lr=0.01)

#os.mkdir(args.directory_path)

# train
model_trainer = DKMTrainer(device, model, optimizer, args, print_freq=2)
print('start training')
model_trainer.train(data_loader, args.epochs, early_stopping=None)

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

plt.show()
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

plot_posterior_2d(test_data_loader, model, 'Posterior', args)


torch.cuda.empty_cache()
