import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
import torch
import gpytorch
from utils import context_target_split
from plotting_functions_DKL import plot_posterior
from torch.utils.data import DataLoader
from dataset_generator import SineData, MultiGPData
from DKModel import GPRegressionModel, DKMTrainer
import os
from training_module import NeuralProcessTrainer
from neural_process import NeuralProcess
from multihead_attention_np import AttentiveNeuralProcess

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
parser.add_argument('--seed', type=int, default=17, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--x-dim', type=int, default=1, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--y-dim', type=int, default=1, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--epochs', type=int, default=100, metavar='G',
                    help='training epochs')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--num-tot-samples', type=int, default=50, metavar='N',
                    help='batch size for np training')
parser.add_argument('--num-context', type=int, default=10, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--num-target', type=int, default=190, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--directory-path', default='/home/francesco/PycharmProjects/MasterThesis/plots/compare/',
                    help='path to plots folder')
parser.add_argument('--early-stopping', type=float, default=0.0, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--print-freq', type=int, default=10, metavar='N',
                    help='dimension of hidden layers in np')

parser.add_argument('--z-dim-dkl', type=int, default=1, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--h-dim-dkl', type=int, default=100, metavar='N',
                    help='dimension of hidden layers in np')

parser.add_argument('--z-dim-np', type=int, default=1, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--h-dim-np', type=int, default=128, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--a-dim-np', type=int, default=256, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--r-dim-np', type=int, default=56, metavar='N',
                    help='dimension of hidden layers in np')

args = parser.parse_args()

## seedings
np.random.seed(args.seed)
torch.manual_seed(args.seed)

########################
#####  Parameters  #####
########################

# dataset parameters
kernel = ['RBF']  # possible kernels ['RBF', 'cosine', 'linear', 'LCM', 'polynomial', 'periodic']
mean = ['linear']  # possible means ['linear', 'constant']
learning_rate = 1e-4
l = '1e-4'
x_range = (-3., 3.)

id = 'DKM_{}fcts_{}e_{}b_{}lr_{}z_{}h__'.format(args.num_tot_samples, args.epochs, args.batch_size,
                                              l, args.z_dim_dkl, args.h_dim_dkl)
id += 'NNP__{}lr_{}z_{}h_{}a_{}r_CinT_'.format(l, args.z_dim_np, args.h_dim_np, args.a_dim_np, args.r_dim_np)+kernel[0]
anp = False

args.directory_path += id
os.mkdir(args.directory_path)

# Create dataset
dataset = MultiGPData(mean, kernel, num_samples=args.num_tot_samples, amplitude_range=x_range, num_points=200)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = MultiGPData(mean, kernel, num_samples=args.batch_size, amplitude_range=x_range, num_points=200)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

for data_init in data_loader:
    break
x_init, y_init = data_init
x_init, y_init, _, _ = context_target_split(x_init[0:1], y_init[0:1], args.num_context, args.num_target)
print('dataset created', x_init.size())

# create model
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model_dkl = GPRegressionModel(x_init, y_init.squeeze(0).squeeze(-1), likelihood,
                          args.h_dim_dkl, args.z_dim_dkl, name_id='DKL').to(device)
if anp:
    model_np = AttentiveNeuralProcess(args.x_dim, args.y_dim, args.r_dim_np, args.z_dim_np, args.h_dim_np, args.a_dim_np,
                                      use_self_att=True, fixed_sigma=None).to(device)
else:
    model_np = NeuralProcess(args.x_dim, args.y_dim, args.r_dim_np, args.z_dim_np, args.h_dim_np, fixed_sigma=None).to(device)

optimizer_dkl = torch.optim.Adam([
    {'params': model_dkl.feature_extractor.parameters()},
    {'params': model_dkl.covar_module.parameters()},
    {'params': model_dkl.mean_module.parameters()},
    {'params': model_dkl.likelihood.parameters()}], lr=0.01)
trainer_dkl = DKMTrainer(device, model_dkl, optimizer_dkl, args, print_freq=args.print_freq)

optimizer_np = torch.optim.Adam(model_np.parameters(), lr=learning_rate)
np_trainer = NeuralProcessTrainer(device, model_np, optimizer_np,
                                  num_context_range=(args.num_context,args.num_context),
                                  num_extra_target_range=(args.num_target,args.num_target),
                                  print_freq=args.print_freq)
# train
print('start dkl training')
t_np_t0 = time.time()
model_np.training = True
np_trainer.train(data_loader, args.epochs, early_stopping=args.early_stopping)
t_np_t1 = time.time()

t_dkl_t0 = time.time()
trainer_dkl.train_dkl(data_loader, args.epochs, early_stopping=args.early_stopping)
t_dkl_t1 = time.time()



# Visualize data samples
plt.figure(1)
plt.title('Samples from gp with kernels: ' + ' '.join(kernel))
for i in range(args.num_tot_samples):
    x, y = dataset[i]
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), c='b', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_range[0], x_range[1])
plt.savefig(args.directory_path + '/'+'-'.join(kernel) + '_data')
plt.close()

if args.epochs > 1:
    plt.figure(2)
    plt.title('Average loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('average loss')
    plt.plot(np.arange(0, len(trainer_dkl.epoch_loss_history)), trainer_dkl.epoch_loss_history, c='r', alpha=0.5,
             label='DKL, time: {:.2f}s'.format(t_dkl_t1-t_dkl_t0))
    plt.plot(np.arange(0, len(np_trainer.epoch_loss_history)), np_trainer.epoch_loss_history, c='b', alpha=0.5,
             label='ANP, time: {:.2f}s'.format(t_np_t1-t_np_t0))
    plt.grid()
    plt.legend()
    plt.savefig(args.directory_path + '/_loss_history_'+id)
    plt.close()

x_target = torch.linspace(x_range[0]-1, x_range[1]+1, 100)
x_target = x_target.unsqueeze(1).unsqueeze(0)

plt.figure(4)
colors = ['r', 'b', 'g', 'y']
num_func = 1
for j in range(num_func):
    for data_test in test_data_loader:
        break
    x, y = data_test

    x_context, y_context, x_target, y_target = context_target_split(x[0:1], y[0:1],
                                                                    args.num_context,
                                                                    args.num_target)
    plt.xlabel('x')
    plt.ylabel('means of y distribution')
    plt.title('Posterior distributions')
    plt.scatter(x_context[0].cpu().numpy(), y_context[0].cpu().numpy(), c='k', label='Context points')
    plt.plot(x[0].cpu().numpy(), y[0].cpu().numpy(), alpha=0.5, c='k', label='Real function')
    ## DKL
    model_dkl.set_train_data(x_context, y_context.squeeze(0).squeeze(-1), strict=False)
    model_dkl.training = False
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False):
        p_y_pred = model_dkl(x[0:1].unsqueeze(0))
    # Extract mean of distribution
    mu = p_y_pred.loc.view(-1).detach().cpu().numpy()
    stdv = p_y_pred.stddev.view(-1).detach().cpu().numpy()
    plt.plot(x[0:1].view(-1).cpu().numpy(), mu,  alpha=0.9, c='r', label='DKL posterior')
    plt.fill_between(x[0:1].view(-1).cpu().numpy(), mu - stdv, mu + stdv, color='r', alpha=0.1)


    ## NP
    model_np.training = False
    for i in range(1):
        # Neural process returns distribution over y_target
        p_y_pred = model_np(x_context, y_context, x)
        # Extract mean of distribution
        mu = p_y_pred.loc.view(-1).detach().cpu().numpy()
        stdv = p_y_pred.stddev.view(-1).detach().cpu().numpy()
        plt.plot(x.view(-1).cpu().numpy(), mu, alpha=0.9, c='b', label='ANP posterior')
        plt.fill_between(x[0:1].view(-1).cpu().numpy(), mu - stdv, mu + stdv, color='b', alpha=0.1)

plt.legend()
plt.savefig(args.directory_path + '/' + id)
plt.close()

torch.cuda.empty_cache()
