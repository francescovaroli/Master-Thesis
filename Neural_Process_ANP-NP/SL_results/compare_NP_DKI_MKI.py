import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
import torch
import gpytorch
from utils.utils import context_target_split_CinT
from torch.utils.data import DataLoader
from utils.dataset_generator import MultiGPData
from DKModel import GPRegressionModel, DKMTrainer
import os
from training_module import NeuralProcessTrainer
from neural_process import NeuralProcess
from previous_methods.attentive_neural_process import AttentiveNeuralProcess
from MeanInterpolatorModel import MeanInterpolator, MITrainer

if torch.cuda.is_available() and False:
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')
print('device: ', device)

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--seed', type=int, default=17, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--directory-path', default='/home/francesco/PycharmProjects/MasterThesis/plots/compare/np_dkl_mi',
                    help='path to plots folder')
parser.add_argument('--print-freq', type=int, default=10, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')

parser.add_argument('--x-dim', type=int, default=1, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--y-dim', type=int, default=1, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--num-tot-samples', type=int, default=100, metavar='N',
                    help='batch size for np training')
parser.add_argument('--mean', default=['constant'],
                    help='dataset mean')
parser.add_argument('--kernel', default=['RBF'],
                    help='dataset kernel')
parser.add_argument('--x-range', default=[(-2., 2.)],
                    help='tested range')
parser.add_argument('--num-points', default=100,
                    help='tested range')

parser.add_argument('--context-range', default=(10, 50), metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--test-context', type=int, default=20, metavar='N',
                    help='dimension of latent variable in np')

#  ##  NP
parser.add_argument('--epochs-np', type=int, default=300, metavar='G',
                    help='training epochs')
parser.add_argument('--z-dim-np', type=int, default=128, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--h-dim-np', type=int, default=128, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--a-dim-np', type=int, default=64, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--r-dim-np', type=int, default=128, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--early-stopping', type=float, default=0.0, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--use-attention', default=True, metavar='N',
                    help='ANP ')

#  ##  DKL
parser.add_argument('--z-dim-dkl', type=int, default=1, metavar='N',
                    help='dimension of latent variable ')

parser.add_argument('--h-dim-dkl', type=int, default=256, metavar='N',
                    help='dimension of hidden layers ')
parser.add_argument('--epochs-dkl', type=int, default=1, metavar='G',
                    help='training epochs')

#  ##  MI
parser.add_argument('--z-dim-mi', type=int, default=3, metavar='N',
                    help='dimension of latent variable ')
parser.add_argument('--h-dim-mi', type=int, default=256, metavar='N',
                    help='dimension of proj layer')
parser.add_argument('--epochs-mi', type=int, default=100, metavar='G',
                    help='training epochs')

args = parser.parse_args()
learning_rate = 5e-4

args.x_range = args.x_range*args.x_dim

#  # Create dataset
if args.x_dim == 1:
    dataset = MultiGPData(args.mean, args.kernel, num_samples=args.num_tot_samples, amplitude_range=args.x_range[0], num_points=args.num_points)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = MultiGPData(args.mean, args.kernel, num_samples=1, amplitude_range=[v*2 for v in args.x_range[0]], num_points=args.num_points)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


#  # Create models

# NP
if args.use_attention:
    model_np = AttentiveNeuralProcess(args.x_dim, args.y_dim, args.r_dim_np, args.z_dim_np, args.h_dim_np, args.a_dim_np, att_type='multihead').to(device)
else:
    model_np = NeuralProcess(args.x_dim, args.y_dim, args.r_dim_np, args.z_dim_np, args.h_dim_np).to(device)
optimizer_np = torch.optim.Adam(model_np.parameters(), lr=learning_rate)
np_trainer = NeuralProcessTrainer(device, model_np, optimizer_np,
                                  num_context_range=args.context_range,
                                  num_extra_target_range=args.num_points,
                                  print_freq=5040)

# MI
model_mi = MeanInterpolator(1, args.h_dim_mi, args.z_dim_mi).to(device).double()
optimizer_mi = torch.optim.Adam([
    {'params': model_mi.feature_extractor.parameters(), 'lr': learning_rate},
    {'params': model_mi.interpolator.parameters(), 'lr': learning_rate}])
trainer_mi = MITrainer(device, model_mi, optimizer_mi, num_context=args.test_context,
                       num_target=args.num_points-args.test_context, print_freq=10)

# DKL
for data_init in data_loader:
    break
x_init, y_init = data_init
x_init, y_init, _, _ = context_target_split_CinT(x_init[0:1], y_init[0:1], args.test_context, args.num_points-args.test_context)

likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)  # noise_constraint=gpytorch.constraints.GreaterThan(5e-2)
model_dkl = GPRegressionModel(x_init, y_init.squeeze(0).squeeze(-1), likelihood,
                              args.h_dim_dkl, args.z_dim_dkl, name_id='DKL').to(device)
optimizer_dkl = torch.optim.Adam([
    {'params': model_dkl.feature_extractor.parameters()},
    {'params': model_dkl.covar_module.parameters()},
    {'params': model_dkl.mean_module.parameters()},
    {'params': model_dkl.likelihood.parameters()}], lr=0.05)
trainer_dkl = DKMTrainer(device, model_dkl, optimizer_dkl, num_context=args.test_context,
                         num_target=args.num_points-args.test_context, print_freq=args.print_freq)


#  Visualize data samples
nm = time.ctime()
args.directory_path += nm
os.mkdir(args.directory_path)
plt.figure(1)
plt.title('Samples from GP with {} kernel'.format(args.kernel[0]))
for i in range(args.num_tot_samples):
    x, y = dataset[i]
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), c='b', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(args.x_range[0][0], args.x_range[0][1])

plt.savefig(args.directory_path + '/_dataset')
plt.close()


#  #  Train

#  NP
t_np_t0 = time.time()
model_np.training = True
np_trainer.train(data_loader, args.epochs_np, early_stopping=args.early_stopping)
t_np_t1 = time.time()

#  DKL
print('start training')
model_dkl.train()
likelihood.train()
t_dkl_t0 = time.time()
try:
    trainer_dkl.train(data_loader, args.epochs_dkl, early_stopping=args.early_stopping)
except RuntimeError:
    pass
t_dkl_t1 = time.time()
model_dkl.eval()
likelihood.eval()

# MI
t_mi_t0 = time.time()
trainer_mi.train(data_loader, args.epochs_mi, early_stopping=None)
t_mi_t1 = time.time()


# plot loss
fig_loss = plt.figure(figsize=(14, 10))
ax_loss = fig_loss.add_subplot(111)
ax_loss.set_title('Average loss over epochs')
ax_loss.set_xlabel('Epochs')
ax_loss.set_ylabel('Average loss')
colors = ['b', 'r', 'g']
labels = [', L := -log_likelihood + kl', ', L := -log_likelihood', ', L := euclidian_dist']
for i, trainer in enumerate([np_trainer, trainer_dkl, trainer_mi]):
    try:
        ax_loss.plot(np.arange(len(trainer.epoch_loss_history)), trainer.epoch_loss_history, c=colors[i], label=trainer.model.id + labels[i])
    except AttributeError:
                ax_loss.plot(np.arange(len(trainer.epoch_loss_history)), trainer.epoch_loss_history, c=colors[i], label=trainer.neural_process.id + labels[i])

plt.grid()
plt.legend()
fig_loss.savefig(args.directory_path + '/_loss_history_')
plt.close()


#  #  Evaluation

x_target = torch.linspace(args.x_range[0][0]-1, args.x_range[0][1]+1, 100)
for data_test in test_data_loader:
    break
x, y = data_test
x_context, y_context, _, _ = context_target_split_CinT(x[0:1], y[0:1], args.test_context, 0)

# NP
model_np.training = False
p_y_pred_np = model_np(x_context, y_context, x)
mu_np = p_y_pred_np.loc.view(-1).detach().cpu().numpy()
stdv_np = p_y_pred_np.stddev.view(-1).detach().cpu().numpy()

# DKL
model_dkl.set_train_data(x_context, y_context.squeeze(0).squeeze(-1), strict=False)
model_dkl.training = False
with torch.no_grad(), gpytorch.settings.use_toeplitz(False):
    p_y_pred_dkl = model_dkl(x[0:1].unsqueeze(0))
mu_dkl = p_y_pred_dkl.loc.view(-1).detach().cpu().numpy()
stdv_dkl = p_y_pred_dkl.stddev.view(-1).detach().cpu().numpy()

# MI
mu_mi = model_mi(x_context, y_context, x)

# plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)
ax.set_title('Posterior distributions')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.scatter(x_context[0].cpu().numpy(), y_context[0].cpu().numpy(), c='k', label='Context points')
plt.plot(x[0].cpu().numpy(), y[0].cpu().numpy(), alpha=0.5, c='k', label='Real function')

ax.plot(x.view(-1).cpu().numpy(), mu_np, alpha=0.9, c='b', label=model_np.id+' posterior')
ax.fill_between(x[0:1].view(-1).cpu().numpy(), mu_np - stdv_np, mu_np + stdv_np, color='b', alpha=0.1)

ax.plot(x[0:1].view(-1).cpu().numpy(), mu_dkl, alpha=0.9, c='r', label='DKL posterior')
ax.fill_between(x[0:1].view(-1).cpu().numpy(), mu_dkl - stdv_dkl, mu_dkl + stdv_dkl, color='r', alpha=0.1)

ax.plot(x[0:1].cpu().detach().numpy()[0].squeeze(-1), mu_mi.cpu().detach(), alpha=0.9, c='g', label='Mean Interpolation')
plt.legend()

fig.savefig(args.directory_path + '/posterior')
