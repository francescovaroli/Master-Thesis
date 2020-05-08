import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from training_module import NeuralProcessTrainer
from neural_process import NeuralProcess
from multihead_attention_np import AttentiveNeuralProcess
from utils.dataset_generator import SineData, MultiGPData
from utils.utils import context_target_split
import os
import time
import torch

if torch.cuda.is_available() and False:
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
else:
    device = torch.device('cpu')
print('device: ', device)

plots_path = '/home/francesco/PycharmProjects/MasterThesis/plots/NP&ANP/1D/w-o self attention/'

## seedingsplt
seed = 75
np.random.seed(seed)
torch.manual_seed(seed)

########################
#####  Parameters  #####
########################

# dataset parameters
data = 'gp'
kernel = ['RBF']  # possible kernels ['RBF', 'cosine', 'linear', 'LCM', 'polynomial', 'periodic']
mean = ['constant']  # possible means ['linear', 'constant']
num_tot_samples = 1000

att_type = 'multihead'  # attention_types = ['uniform','laplace','dot_product']

epochs = 500
learning_rate = 3e-4
l = '3e-4'
batch_size = 8
num_context = (20, 30)
num_target = (50, 70)

x_range = (-3., 3.)

x_dim = 1
y_dim = 1
r_dim = 128  # Dimension of representation of context points in NP
z_dim = 2*64  # Dimension of sampled latent variable
h_dim = 128  # Dimension of hidden layers in encoder and decoder
a_dim = 2*64  # Dimension of attention output



# Create dataset
if data == 'sine':
    dataset = SineData(amplitude_range=(-5., 5.),
                       shift_range=(-.5, .5),
                       num_samples=num_tot_samples)
elif data == 'gp':

    dataset = MultiGPData(mean, kernel, num_samples=num_tot_samples, amplitude_range=x_range)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Extract a batch from data_loader
test_x_context_l = []
test_y_context_l = []
test_x_l = []
test_y_l = []
for e, batch in enumerate(data_loader):
# Use batch to create random set of context points
    test_x, test_y = batch
    test_x_l.append(test_x)
    test_y_l.append(test_y)
    num_context_t = 21
    num_target_t = 21
    test_x_context, test_y_context, _, _ = context_target_split(test_x[0:1], test_y[0:1],
                                                                num_context_t,
                                                                num_target_t)
    test_x_context_l.append(test_x_context)
    test_y_context_l.append(test_y_context)

# Visualize data samples
fig_data, ax_data = plt.subplots(1, 1)

#ax_data.set_title('Multiple samples from the GP with ' + kernel[0] + ' kernel')
for i in range(64):
    x, y = dataset[i*(num_tot_samples//64)]
    ax_data.plot(x.cpu().numpy(), y.cpu().numpy(), c='k', alpha=0.5)
    ax_data.set_xlabel('x')
    ax_data.set_ylabel('y')
    ax_data.set_xlim(x_range[0], x_range[1])
fig_data.savefig(plots_path + '-'.join(kernel) + '_data', dpi=250)
plt.close(fig_data)

fig_epoch, ax_epoch = plt.subplots(1, 1)
#ax_epoch.set_title('Average loss over epochs')
ax_epoch.set_xlabel('Epochs')
ax_epoch.set_ylabel('Average loss')
ax_epoch.grid()

use_self_att = False
first = True
for use_attention in [ False]:#[False]:False, True,
    if use_attention:
        if not first:
            break
            use_self_att = True

    # create ID for saving plots
    print('attention:' ,use_attention)
    print('self attention:', use_self_att)
    mdl = 'NP'
    color = 'b'
    if use_attention:
        mdl = 'cross attention '+mdl
        color = 'r'
        if use_self_att:
            mdl = 'self & ' + mdl
            color = 'g'

    id = mdl +time.ctime()+ '{}e_{}b_{}c{}t_{}lr_{}r_{}z_{}a'.format(epochs, batch_size, num_context, num_target, l,r_dim, z_dim, a_dim)
    # create and train np
    if use_attention:
        neuralprocess = AttentiveNeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim, a_dim, use_self_att=use_self_att).to(device)
        first = False
    else:
        neuralprocess = NeuralProcess(x_dim, y_dim, r_dim, z_dim, h_dim).to(device)

    t0 = time.time()
    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=learning_rate)
    np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                      num_context_range=num_context,
                                      num_extra_target_range=num_target,
                                      print_freq=50000)
    neuralprocess.training = True
    np_trainer.train(data_loader, epochs, early_stopping=0)

    '''plot training epochs'''
    n_ep = len(np_trainer.epoch_loss_history)
    ax_epoch.plot(np.linspace(0, n_ep-1 ,n_ep), np_trainer.epoch_loss_history, c=color, label=mdl)

    x_target = torch.linspace(x_range[0], x_range[1], 100)
    x_target = x_target.unsqueeze(1).unsqueeze(0)

    # plot prior
    if False:
        fig_prior, ax_prior = plt.subplots(1, 1)
        ax_prior.set_ylabel('Means of y distribution')
        ax_prior.set_xlabel('x')
        #ax_prior.set_title('Samples from prior distribution')
        for i in range(60):
            z_sample = torch.randn((1, z_dim))
            z_sample = z_sample.unsqueeze(1).repeat(1, 100, 1)
            mu, _ = neuralprocess.xz_to_y(x_target, z_sample)
            ax_prior.plot(x_target.cpu().numpy()[0], mu.cpu().detach().numpy()[0],
                     c='b', alpha=0.5)
        fig_prior.savefig(plots_path + '-'.join(kernel) + '_prior_'+id, dpi=250)
        plt.close(fig_prior)



    neuralprocess.training = False
    for l in range(3):
        test_x_context = test_x_context_l[l]
        test_y_context = test_y_context_l[l]
        fig_post = plt.figure(figsize=plt.figaspect(2))
        ax_post = fig_post.add_subplot(2, 1, 1)
        fig_post.tight_layout()

        ax_post.set_title('Multiple predictions of the mean')
        realizations = torch.zeros_like(test_x[0])
        num_realizations = 50
        for i in range(num_realizations-1):
            # Neural process returns distribution over y_target
            p_y_pred = neuralprocess(test_x_context, test_y_context, x_target)
            # Extract mean of distribution
            ax_post.set_xlabel('x')
            ax_post.set_ylabel('means of y distribution')
            mu = p_y_pred.loc.detach()
            std = p_y_pred.stddev.detach()
            ax_post.plot(x_target.cpu().numpy()[0], mu.cpu().numpy()[0],
                     alpha=0.05, c=color)
        ax_post.plot(x_target.cpu().numpy()[0], mu.cpu().numpy()[0],
                     alpha=0.05, c=color, label='Predicted means')
        ax_post.plot(test_x_l[l][0].cpu().numpy(), test_y_l[l][0].cpu().numpy(), alpha=0.3, c='k', label='Real function')
        ax_post.scatter(test_x_context[0].cpu().numpy(), test_y_context[0].cpu().numpy(), c='k', label='Context points')
        ax_post.legend()
        ax_post_avg = fig_post.add_subplot(2, 1, 2)
        ax_post_avg.plot(x_target.cpu().numpy()[0], mu.cpu().numpy()[0],
                     alpha=0.5, c=color, label='Predicted mean')
        std_h = mu.cpu() + std.cpu()
        std_l = mu.cpu() - std.cpu()
        ax_post_avg.fill_between(x_target[0,:,0].cpu(), std_l[0,:,0], std_h[0,:,0] ,alpha=0.05, color=color, label='Standard deviation')
        ax_post_avg.plot(test_x_l[l][0].cpu().numpy(), test_y_l[l][0].cpu().numpy(), alpha=0.15, c='k')
        ax_post_avg.scatter(test_x_context[0].cpu().numpy(), test_y_context[0].cpu().numpy(), c='k')
        ax_post_avg.legend()
        ax_post_avg.set_title('Single posterior distribution')
        ax_post_avg.set_xlabel('x')
        ax_post_avg.set_ylabel('y distribution')

        fig_post.savefig(plots_path + '_posterior_' + str(l) + id)
        plt.show()
        plt.close(fig_post)
    print(mdl + ' duration:', (time.time()-t0)/60, ' minutes')
    torch.cuda.empty_cache()

leg = ax_epoch.legend(loc="upper right")
fig_epoch.savefig(plots_path + '_loss_history_'+id, dpi=250)