import torch
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from torch.distributions.kl import kl_divergence
from utils.utils import (context_target_split_CinT)


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()

def plot_functions(target_x, target_y, context_x, context_y, pred_y, std):
    """Plots the predicted mean and variance and the context points.

    Args:
      target_x: An array of shape [B,num_targets,1] that contains the
          x values of the target points.
      target_y: An array of shape [B,num_targets,1] that contains the
          y values of the target points.
      context_x: An array of shape [B,num_contexts,1] that contains
          the x values of the context points.
      context_y: An array of shape [B,num_contexts,1] that contains
          the y values of the context points.
      pred_y: An array of shape [B,num_targets,1] that contains the
          predicted means of the y values at the target points in target_x.
      std: An array of shape [B,num_targets,1] that contains the
          predicted std dev of the y values at the target points in target_x.
    """
    # Plot everything
    xt, yp = zip(*sorted(zip(target_x[0].numpy(), pred_y[0].numpy())))
    _, yt = zip(*sorted(zip(target_x[0].numpy(), target_y[0].numpy())))
    xc, yc = zip(*sorted(zip(context_x[0].numpy(), context_y[0].numpy())))
    fig_train, ax_train = plt.subplots(1, 1)
    ax_train.plot(xt, yp, 'b', linewidth=2)
    ax_train.plot(xt, yt, 'k:', linewidth=2)
    ax_train.plot(xc, yc, 'ko', markersize=10)
    _, stdm = zip(*sorted(zip(target_x[0].numpy(), pred_y[0, :] - std[0,:])))
    _, stdp = zip(*sorted(zip(target_x[0].numpy(), pred_y[0, :] + std[0,:])))
    ax_train.fill_between(
        [xt[i][0] for i in range(len(xt))], stdm, stdp,
        alpha=0.2,
        facecolor='#65c9f7',
        interpolate=True)

    # Make the plot pretty
    plt.grid('off')
    #plt.show()
    plt.close(fig_train)

class NeuralProcessTrainer():
    """
    Class to handle training of Neural Processes and Attentive Neural Process
    for functions and images.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or neural_process.AttentiveNeuralProcess
                     or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    num_extra_target_range : tuple of ints
        Number of extra target points (as we always include context points in
        target points, i.e. context points are a subset of target points) will
        be sampled uniformly in the range given by num_extra_target_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 num_extra_target_range=100, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq

        # Check if neural process is for images
        self.steps = 0
        self.epoch_loss_history = []

    def train(self, data_loader, epochs, early_stopping=None):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                # Sample number of context and target points
                num_context = randint(*self.num_context_range)

                # Create context and target points and apply neural process

                x, y = data

                x_context, y_context, _, _ = \
                    context_target_split_CinT(x, y, num_context, 0)
                p_y_pred, q_target, q_context = \
                    self.neural_process(x_context, y_context, x, y)

                loss = self._loss(p_y_pred, y, q_target, q_context)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                self.steps += 1

            avg_loss = epoch_loss / len(data_loader)
            print("Epoch: {}, Avg_loss: {}".format(epoch, avg_loss))
            self.epoch_loss_history.append(avg_loss)

            if early_stopping is not None:
                if avg_loss < early_stopping:
                    break

    def _loss(self, p_y_pred, y_target, q_target, q_context):
        """
        Computes Neural Process loss.

        Parameters
        ----------
        p_y_pred : one of torch.distributions.Distribution
            Distribution over y output by Neural Process.

        y_target : torch.Tensor
            Shape (batch_size, num_target, y_dim)

        q_target : one of torch.distributions.Distribution
            Latent distribution for target points.

        q_context : one of torch.distributions.Distribution
            Latent distribution for context points.
        """
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl
