import torch
from random import randint
from torch.distributions.kl import kl_divergence
from utils_rl.memory_dataset import merge_context, get_random_context
from utils.utils import context_target_split_CinT, context_target_split
import time


class NeuralProcessTrainerRL():
    """
    Class to handle training of Neural Processes and Attentive Neural Process as component of IMeL.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or neural_process.AttentiveNeuralProcess
                     or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    num_context_range : tuple of ints
        Number of context points will be sampled uniformly in the range given
        by num_context_range.

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_context_range,
                 print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.num_context_range = num_context_range
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
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                self.optimizer.zero_grad()

                x, y, num_points = data
                num_points = min(num_points).item()
                # Sample number of context and target points
                num_context = min(randint(*self.num_context_range), num_points//2)
                num_extra_target = num_points-num_context

                # Create context sampling the RM
                x_context, y_context, x_target, y_target = \
                    context_target_split_CinT(x[:, :num_points,:], y[:, :num_points, :], num_context, num_extra_target)
                p_y_pred, q_target, q_context = \
                    self.neural_process(x_context, y_context, x_target, y_target)
                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                self.steps += 1
            avg_loss = epoch_loss / len(data_loader)
            if epoch % self.print_freq == 0 or epoch == epochs-1:
                print("Epoch rl: {}, Avg_loss: {}".format(epoch, avg_loss))
            self.epoch_loss_history.append(avg_loss)

            if early_stopping is not None:
                if avg_loss < early_stopping:
                    break
            torch.cuda.empty_cache()

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

class NeuralProcessTrainerLoo():
    """
    Class to handle training of Neural Processes and Attentive Neural Process
     as component of IMeL. Leave-one-out training.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or neural_process.AttentiveNeuralProcess
                     or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers


    num_target : num of points in the target set

    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, num_target=100, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.print_freq = print_freq
        self.steps = 0
        self.epoch_loss_history = []
        self.num_target = num_target

    def train(self, data_loader, epochs, early_stopping=None):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """

        # compute the episode-specific context sets
        one_out_list = []
        episode_fixed_list = [ep for _, ep in enumerate(data_loader)]
        for i in range(len(episode_fixed_list)):
            context_list = []
            if len(episode_fixed_list) == 1:
                context_list = [ep for ep in episode_fixed_list]
            else:
                for j, ep in enumerate(episode_fixed_list):
                    if j != i:
                        context_list.append(ep)
                #context_list = [ep for j, ep in enumerate(data_loader) if j != i]
            all_context_points = merge_context(context_list)
            one_out_list.append(all_context_points)

        for epoch in range(epochs):
            epoch_loss = 0.
            for i in range(len(data_loader)):
                self.optimizer.zero_grad()

                all_context_points = one_out_list[i]
                data = episode_fixed_list[i]
                x, y, num_points = data
                num_target = min(num_points.item(), self.num_target)
                x_context, y_context = all_context_points
                _, _, x_target, y_target = context_target_split(x[:, :num_points, :], y[:, :num_points, :],
                                                                0, num_target)
                p_y_pred, q_target, q_context = \
                    self.neural_process(x_context, y_context, x_target, y_target)
                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                self.steps += 1
            avg_loss = epoch_loss / len(data_loader)
            if epoch % self.print_freq == 0 or epoch == epochs-1:
                print("Epoch loo: {}, Avg_loss: {}".format(epoch, avg_loss))
            self.epoch_loss_history.append(avg_loss)

            if early_stopping is not None:
                if avg_loss < early_stopping:
                    break
            #torch.cuda.empty_cache()
    def _loss(self, p_y_pred, y_target, q_target, q_context):

        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl



class NeuralProcessTrainerLooPick():
    """
    Class to handle training of Neural Processes and Attentive Neural Process
    as component of IMeL. Leave-one-out training choosing the context set depending on the index/distance.

    Parameters
    ----------
    device : torch.device

    neural_process : neural_process.NeuralProcess or neural_process.AttentiveNeuralProcess
                     or NeuralProcessImg instance

    optimizer : one of torch.optim optimizers

    pick_dist : distance value (None if index is used)

    num_context : num samples selected


    print_freq : int
        Frequency with which to print loss information during training.
    """
    def __init__(self, device, neural_process, optimizer, pick_dist, num_context, print_freq=100):
        self.device = device
        self.neural_process = neural_process
        self.optimizer = optimizer
        self.print_freq = print_freq
        self.steps = 0
        self.epoch_loss_history = []
        self.num_context = num_context
        self.pick_dist = pick_dist

    def train(self, data_loader, epochs, early_stopping=None):
        """
        Trains Neural Process.

        Parameters
        ----------
        dataloader : torch.utils.DataLoader instance

        epochs : int
            Number of epochs to train for.
        """
        # compute context sets
        one_out_list = []
        episode_fixed_list = [ep for _, ep in enumerate(data_loader)]
        for i in range(len(episode_fixed_list)):
            context_list = []
            if len(episode_fixed_list) == 1:
                context_list = [ep for ep in episode_fixed_list]
            else:
                for j, ep in enumerate(episode_fixed_list):
                    if j != i:
                        context_list.append(ep)
            one_out_list.append(context_list)

        for epoch in range(epochs):
            epoch_loss = 0.
            for i in range(len(data_loader)):
                self.optimizer.zero_grad()

                all_context_points = one_out_list[i]
                data = episode_fixed_list[i]
                x, y, num_points = data

                x_context, y_context = get_random_context(all_context_points, self.num_context)
                #x_context, y_context = get_close_context(index, context_list, num_tot_context=self.num_context)
                x_target = x[:, :num_points, :]
                y_target = y[:, :num_points, :]
                p_y_pred, q_target, q_context = \
                    self.neural_process(x_context, y_context, x_target, y_target)
                loss = self._loss(p_y_pred, y_target, q_target, q_context)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                self.steps += 1
            avg_loss = epoch_loss / len(data_loader)
            if epoch % self.print_freq == 0 or epoch == epochs-1:
                print("Epoch pick: {}, Avg_loss: {}".format(epoch, avg_loss))
            self.epoch_loss_history.append(avg_loss)

            if early_stopping is not None:
                if avg_loss < early_stopping:
                    break
            #torch.cuda.empty_cache()

    def _loss(self, p_y_pred, y_target, q_target, q_context):
        # Log likelihood has shape (batch_size, num_target, y_dim). Take mean
        # over batch and sum over number of targets and dimensions of y
        log_likelihood = p_y_pred.log_prob(y_target).mean(dim=0).sum()
        # KL has shape (batch_size, r_dim). Take mean over batch and sum over
        # r_dim (since r_dim is dimension of normal distribution)
        kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
        return -log_likelihood + kl


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])