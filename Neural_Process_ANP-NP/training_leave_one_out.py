import torch
from random import randint
import random
from utils_rl.memory_dataset import merge_context
from torch.distributions.kl import kl_divergence
import time
import numpy as np

class NeuralProcessTrainerLoo():
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
                 num_extra_target_range, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
        self.num_extra_target_range = num_extra_target_range
        self.print_freq = print_freq
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
        one_out_list = []
        for i in range(len(data_loader)):
            if len(data_loader) == 1:
                context_list = [ep for j, ep in enumerate(data_loader)]
            else:
                context_list = [ep for j, ep in enumerate(data_loader) if j != i]
            all_context_points = merge_context(context_list)
            one_out_list.append(all_context_points)

        for epoch in range(epochs):
            epoch_loss = 0.
            for i in range(len(data_loader)):
                self.optimizer.zero_grad()

                all_context_points = one_out_list[i]
                data = data_loader.dataset[i]
                x, y, num_points = data
                x_context, y_context = all_context_points
                index = random.randint(0, num_points)
                x_target = x[index].unsqueeze(0).unsqueeze(1)
                y_target = y[index].unsqueeze(0).unsqueeze(1)
                p_y_pred, q_target, q_context = \
                    self.neural_process(x_context, y_context, x_target, y_target)
                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                self.steps += 1
            avg_loss = epoch_loss / len(data_loader)
            if epoch % self.print_freq == 0 or epoch == epochs-1:
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
