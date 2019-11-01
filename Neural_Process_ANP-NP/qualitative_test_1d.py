import numpy as np
import matplotlib.pyplot as plt
import torch
from random import randint
from torch.utils.data import DataLoader
from training_module import NeuralProcessTrainer
from neural_process import NeuralProcess
from multihead_attention_np import AttentiveNeuralProcess
from dataset_generator import SineData, MultiGPData
from utils import context_target_split
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
else:
    device = torch.device('cpu')
print('device: ', device)

plots_path = '/home/francesco/PycharmProjects/MasterThesis/plots/NP&ANP/1D/w-o self attention/'

## seedings
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

########################
#####  Parameters  #####
########################

# dataset parameters
data = 'gp'
kernel = ['matern']  # possible kernels ['RBF', 'cosine', 'linear', 'LCM', 'polynomial', 'periodic']
mean = ['linear']  # possible means ['linear', 'constant']
num_tot_samples = 2000
use_different_test_dataset = False

# model parameters
use_attention = False
use_self_att = True
att_type = 'dot_product'  # attention_types = ['uniform','laplace','dot_product']

epochs = 2
learning_rate = 3e-4
l = '3e-4'
batch_size = 4
num_context = (10, 20)
num_target = (25, 50)

x_range = (-3., 3.)

x_dim = 1
y_dim = 1
r_dim = 128  # Dimension of representation of context points in NP
z_dim = 128  # Dimension of sampled latent variable
h_dim = 2*128  # Dimension of hidden layers in encoder and decoder
a_dim = 128  # Dimension of attention output

# create ID for saving plots
mdl = 'NP'
if use_attention:
    mdl = 'cross attention '+mdl
if use_self_att:
    mdl = 'self & ' + mdl

id = mdl + '{}e_{}b_{}c{}t_{}lr_{}r_{}z_{}a'.format(epochs, batch_size, num_context, num_target, l,r_dim, z_dim, a_dim)


# Create dataset
if data == 'sine':
    dataset = SineData(amplitude_range=(-5., 5.),
                       shift_range=(-.5, .5),
                       num_samples=num_tot_samples)
elif data == 'gp':

    dataset = MultiGPData(mean, kernel, num_samples=num_tot_samples, amplitude_range=x_range)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if use_different_test_dataset:
        test_dataset = MultiGPData(mean, kernel, num_samples=batch_size, amplitude_range=x_range)
        test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    else:
        test_data_loader = data_loader

# Visualize data samples
plt.figure(1)

plt.title('Samples from '+data+' with kernels: ' + ' '.join(kernel))
for i in range(64):
    x, y = dataset[i*(num_tot_samples//64)]
    plt.plot(x.cpu().numpy(), y.cpu().numpy(), c='b', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_range[0], x_range[1])
plt.savefig(plots_path + '-'.join(kernel) + '_data')
plt.close()
# create and train np
if use_attention:
    neuralprocess = AttentiveNeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim, a_dim, use_self_att=use_self_att).to(device)
else:
    neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim).to(device)


optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=learning_rate)
np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                  num_context_range=num_context,
                                  num_extra_target_range=num_target,
                                  print_freq=10000)
neuralprocess.training = True
np_trainer.train(data_loader, epochs)

plt.figure(2)
plt.title('average loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('average loss')
plt.plot(np.linspace(0, epochs-1 ,epochs), np_trainer.epoch_loss_history, c='b', alpha=0.5)
plt.grid()
plt.savefig(plots_path + '_loss_history_'+id)
plt.close()

x_target = torch.linspace(x_range[0]-1, x_range[1]+1, 100)
x_target = x_target.unsqueeze(1).unsqueeze(0)

if not use_attention:
    for i in range(60):
        z_sample = torch.randn((1, z_dim))
        z_sample = z_sample.unsqueeze(1).repeat(1, 100, 1)
        mu, _ = neuralprocess.xz_to_y(x_target, z_sample)
        plt.figure(3)
        plt.xlabel('x')
        plt.ylabel('means of y distribution')
        plt.title('Samples from trained prior')
        plt.plot(x_target.cpu().numpy()[0], mu.cpu().detach().numpy()[0],
                 c='b', alpha=0.5)

plt.savefig(plots_path + '-'.join(kernel) + '_prior_'+id)
plt.close()
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

plt.figure(4)
plt.title(mdl + ' Posterior')
for i in range(64):
    # Neural process returns distribution over y_target
    p_y_pred = neuralprocess(x_context, y_context, x_target)
    # Extract mean of distribution
    plt.xlabel('x')
    plt.ylabel('means of y distribution')
    mu = p_y_pred.loc.detach()
    plt.plot(x_target.cpu().numpy()[0], mu.cpu().numpy()[0],
             alpha=0.05, c='b')

plt.plot(x[0].cpu().numpy(), y[0].cpu().numpy(), alpha=0.3, c='k')
plt.scatter(x_context[0].cpu().numpy(), y_context[0].cpu().numpy(), c='k')
plt.savefig(plots_path + '_posterior_'+id)
plt.show()
plt.close()

torch.cuda.empty_cache()
