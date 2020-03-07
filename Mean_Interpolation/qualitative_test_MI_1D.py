import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
import gpytorch
from utils import context_target_split
from plotting_functions_DKL import plot_posterior
from torch.utils.data import DataLoader
from dataset_generator import SineData, MultiGPData
from MeanInterpolatorModel import MeanInterpolator, MITrainer
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
parser.add_argument('--x-dim', type=int, default=1, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--y-dim', type=int, default=1, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--z-dim', type=int, default=3, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--h-dim', type=int, default=256, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--epochs', type=int, default=100, metavar='G',
                    help='training epochs')
parser.add_argument('--scaling', default='uniform', metavar='N',
                    help='z scaling')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--num-tot-samples', type=int, default=1000, metavar='N',
                    help='batch size for np training')
parser.add_argument('--num-context', type=int, default=20, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--num-target', type=int, default=1, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--x-range', default=[-3, 3], metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--directory-path', default='/home/francesco/PycharmProjects/MasterThesis/plots/MI/1D/',
                    help='path to plots folder')
args = parser.parse_args()

## seedings
np.random.seed(args.seed)
torch.manual_seed(args.seed)

########################
#####  Parameters  #####
########################

# dataset parameters
kernel = ['matern']  # possible kernels ['RBF', 'cosine', 'linear', 'LCM', 'polynomial', 'periodic']
mean = ['linear']  # possible means ['linear', 'constant']
learning_rate = 1e-4
l = '3e-4'
x_range = args.x_range

id = 'MI_{}fcts_{}e_{}b_{}lr_{}z_{}h_{}_{}ctxt_2layers_'.format(args.num_tot_samples, args.epochs, args.batch_size,
                                              l, args.z_dim, args.h_dim, args.scaling, args.num_context)

args.directory_path += id


# Create dataset
dataset = MultiGPData(mean, kernel, num_samples=args.num_tot_samples, amplitude_range=x_range)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
#all_dataset = [dataset.data[0][0], dataset.data[0][1]]
#for func in dataset.data[1:]:
#    all_train_dataset = [torch.cat([all_dataset[0], func[0]], dim=0), torch.cat([all_dataset[1], func[1]], dim=0)]
#train_x, train_y = all_train_dataset
test_dataset = MultiGPData(mean, kernel, num_samples=args.batch_size, amplitude_range=x_range)
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
model = MeanInterpolator(1, args.h_dim, args.z_dim).to(device).double()



optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters(), 'lr':learning_rate},
    {'params': model.interpolator.parameters(), 'lr':learning_rate}])

os.mkdir(args.directory_path)

model_trainer = MITrainer(device, model, optimizer, num_context=args.num_context, print_freq=10)
print('start training')
model_trainer.train(data_loader, args.epochs, early_stopping=None)

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
    plt.title('average loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('average loss')
    plt.plot(np.linspace(0, args.epochs-1 ,args.epochs), model_trainer.epoch_loss_history, c='b', alpha=0.5)
    plt.grid()
    plt.savefig(args.directory_path + '/_loss_history_'+id)
    plt.close()

x_target = torch.linspace(x_range[0]-1, x_range[1]+1, 100)
x_target = x_target.unsqueeze(1)

ys = []


plt.figure(4)
plt.xlabel('x')
plt.ylabel('means of y distribution')
colors = ['r', 'b', 'g', 'y']
for j in range(1):

    x, y = test_data_loader.dataset.data[j]
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1],
                                                      args.num_context,
                                                      args.num_target)
    plt.title('Mean prediction')
    for _x1 in x_target:
        pred = model(x_context, y_context, _x1.unsqueeze(0))
        ys.append(pred.view(-1))

    plt.plot(x[0:1].cpu().numpy()[0].squeeze(-1), ys,
             alpha=0.9, c=colors[j])

    plt.plot(x[0].cpu().numpy(), y[0].cpu().numpy(), alpha=0.5, c='k')
    plt.scatter(x_context[0].cpu().numpy(), y_context[0].cpu().numpy(), c=colors[j], label='context')
plt.legend()
plt.savefig(args.directory_path + '/'+id)
plt.close()
torch.cuda.empty_cache()
