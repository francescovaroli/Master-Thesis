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
from MeanInterpolatorModel import MeanInterpolator, MITrainer

if torch.cuda.is_available():
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
parser.add_argument('--directory-path', default='/home/francesco/PycharmProjects/MasterThesis/plots/compare/',
                    help='path to plots folder')
parser.add_argument('--print-freq', type=int, default=10, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')

parser.add_argument('--x-dim', type=int, default=1, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--y-dim', type=int, default=1, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--num-tot-samples', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--mean', default=['constant'],
                    help='dataset mean')
parser.add_argument('--kernel', default=['RBF'],
                    help='dataset kernel')
parser.add_argument('--x-range', default=[(-2., 2.)],
                    help='tested range')
parser.add_argument('--num-points', default=200,
                    help='tested range')


#  ##  NP
parser.add_argument('--epochs-np', type=int, default=100, metavar='G',
                    help='training epochs')
parser.add_argument('--num-context-np', type=int, default=10, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--num-target-np', type=int, default=190, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--z-dim-np', type=int, default=1, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--h-dim-np', type=int, default=128, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--a-dim-np', type=int, default=256, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--r-dim-np', type=int, default=56, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--early-stopping-np', type=float, default=0.0, metavar='N',
                    help='dimension of hidden layers in np')


#  ##  DKL
parser.add_argument('--z-dim-dkl', type=int, default=1, metavar='N',
                    help='dimension of latent variable ')
parser.add_argument('--h-dim-dkl', type=int, default=100, metavar='N',
                    help='dimension of hidden layers ')
parser.add_argument('--epochs-dkl', type=int, default=2, metavar='G',
                    help='training epochs')

#  ##  MI
parser.add_argument('--z-dim-mi', type=int, default=3, metavar='N',
                    help='dimension of latent variable ')
parser.add_argument('--h-dim-mi', type=int, default=256, metavar='N',
                    help='dimension of proj layer')

args = parser.parse_args()

args.x_range = args.x_range*args.x_dim

# Create dataset
if args.x_dim == 1:
    dataset = MultiGPData(args.mean, args.kernel, num_samples=args.num_tot_samples, amplitude_range=args.x_range[0], num_points=args.num_points)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = MultiGPData(args.mean, args.kernel, num_samples=1, amplitude_range=args.x_range[0], num_points=args.num_points)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


