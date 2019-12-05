import numpy as np
import matplotlib.pyplot as plt
import torch
from random import randint
from torch.utils.data import DataLoader
from training_module import NeuralProcessTrainer
from neural_process import NeuralProcess
from torch import nn

from multihead_attention_np import AttentiveNeuralProcess
from dataset_generator import SineData, MultiGPData
from utils import context_target_split
import os



class InitFunc():

    def init_xavier(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def init_normal(m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def init_zero(m):
        if type(m) == nn.Linear:
            torch.nn.init.constant_(m.weight, 0)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def init_kaiming(m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)

    def init_sparse(m):
        if type(m) == nn.Linear:
            torch.nn.init.sparse_(m.weight, 1e-20)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


def test():
    def plot_weights_and_policy(policy, id):
        '''
        Plots the weights in the different layers.
        '''

        named_parameters = policy.named_parameters()
        fig = plt.figure(figsize=(16, 10))
        #fig.suptitle(id , fontsize=18)
        fig.tight_layout()

        ax_w = fig.add_subplot(211)
        ax_w.set_title(id + " Weights initialization", fontsize=18)

        for n, p in named_parameters:
            color = '#%06X' % randint(0, 0xFFFFFF)
            if 'weight' in n and 'layer_norm' not in n:
                try:
                    weights = p.mean(dim=1)
                except:
                    print(n)
                ax_w.plot(np.arange(len(weights)), weights.detach().numpy(),
                          alpha=0.5,  color=color, label=n.replace('.weight', ''))
        ax_w.legend(loc="upper right")
        ax_w.set_xlabel("Weights")
        ax_w.set_ylabel("initialization")
        ax_w.set_ylim(-2, 2)

        ax_policy = fig.add_subplot(212)
        max_std = 0
        min_std = 0
        for z_sample in z_samples:
            mu, sigma = policy.xz_to_y(x, z_sample)
            ax_policy.plot(x[0, :, 0].numpy(), mu[0, :, 0].detach().numpy(), color='b')
            std_h = mu + sigma
            max_std = max(max_std, std_h.max().detach())
            std_l = mu - sigma
            min_std = min(min_std, std_l.min().detach())
            ax_policy.fill_between(x[0, :, 0].detach(), std_l[0, :, 0].detach(), std_h[0, :, 0].detach(), alpha=0.01, color='b')

        ax_policy.set_xlabel('x')
        ax_policy.set_ylabel('y')
        ax_policy.set_ylim(min(min_std, -1), max(max_std, 1))
        ax_policy.set_title('Policies sampled with z ~ N(0,1)')
        plt.grid(True)
        plt.show()
        fig.savefig('/home/francesco/PycharmProjects/MasterThesis/plots/NP&ANP/1D/weights/'+id+'64')

    z_dim = 128
    dims = 128
    num_points = 100

    x = torch.linspace(-1, 1, num_points).unsqueeze(1).unsqueeze(0)

    z_samples =[]
    for n in range(16):
        z_samples.append(torch.randn((1, z_dim*2)).unsqueeze(1).repeat(1, num_points, 1))
    neural_process = AttentiveNeuralProcess(1, 1, dims, z_dim, dims, z_dim)
    plot_weights_and_policy(neural_process, ' default')
    for init_func in [InitFunc.init_xavier, InitFunc.init_normal, InitFunc.init_zero, InitFunc.init_kaiming, InitFunc.init_sparse]:  #
        init_policy = neural_process.apply(init_func)
        plot_weights_and_policy(init_policy, init_func.__name__)
    #plot_weights(neural_process.named_parameters())
#test()