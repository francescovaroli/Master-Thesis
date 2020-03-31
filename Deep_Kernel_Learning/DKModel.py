import torch
import gpytorch
from utils import context_target_split, context_target_split_CinT
from utils_rl import merge_context
import time
import random
from plotting_functions_DKL import plot_posterior, plot_posterior_2d
from utils_rl.memory_dataset import get_close_context
from torch import autograd
from torch.distributions import MultivariateNormal

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, x_dim, h_dim, out_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(x_dim, h_dim))  # default h_dim = 256
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(h_dim, out_dim))


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, h_dim, z_dim, name_id, scaling='uniform', grid_size=None):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        if grid_size is None:
            grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
            print('grid_size:', grid_size)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=z_dim, is_stationary=False), num_dims=z_dim, grid_size=grid_size))  #
        #self.covar_module = gpytorch.kernels.GridInterpolationKernel(gpytorch.kernels.SpectralMixtureKernel(
        #                                      num_mixtures=2, ard_num_dims=z_dim), num_dims=z_dim, grid_size=grid_size)
        self.likelihood = likelihood
        _, num_points, x_dim = train_x.size()
        self.feature_extractor = LargeFeatureExtractor(x_dim, h_dim, z_dim)
        self.scaling = scaling
        self.id = name_id

    def project(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # We're also scaling the features so that they're nice values
        projected_x = self.feature_extractor(x.squeeze(0))

        if self.scaling == 'uniform':
            projected_x = projected_x - projected_x.min(-2)[0]
            z = (projected_x / projected_x.max(-2)[0])*100

        elif self.scaling == 'normal':
            z = (projected_x - projected_x.mean())/projected_x.std()

        elif self.scaling is None:
            z = projected_x

        z[torch.isnan(z)] = 10000
        return z

    def forward(self, x):

        z = self.project(x)
        mean_x = self.mean_module(z)
        covar_x = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKMTrainer():

    def __init__(self, device, model, optimizer, num_context=100, num_target=100, print_freq=100):

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        self.print_freq = print_freq
        self.steps = 0
        self.epoch_loss_history = []
        self.num_context = num_context
        self.num_target = num_target

    def train_rl(self, data_loader, epochs, early_stopping=None):

        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                t = time.time()
                # Zero backprop gradients
                self.optimizer.zero_grad()
                # Get output from model
                x, y, num_points = data  # add , num_points
                # divide context (N-1) and target (1)
                x_context, y_context, x_target, y_target = context_target_split(x[:,:num_points,:], y[:,:num_points,:],
                                                                  num_points.item()-1, 1)

                #num_points = min(num_points).item()
                self.model.set_train_data(inputs=x_context, targets=y_context.view(-1), strict=False)
                self.model.eval()
                self.model.likelihood.eval()
                #try:
                with gpytorch.settings.use_toeplitz(False):
                    predictions = self.model(x_target)
                self.model.train()
                self.model.likelihood.train()

                loss = -self.mll(predictions, y_target.view(-1))
                if torch.isnan(loss):
                    print(loss)
                    self.model.eval()
                    self.model.likelihood.eval()
                    s = self.model(x_target)
                #except RuntimeError:
                #    predictions = self.model(x_target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                avg_loss = epoch_loss / len(data_loader)
            print('epoch %d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (epoch, loss.item(),
                self.model.covar_module.base_kernel.lengthscale.item(),
                #self.model.covar_module.base_kernel.outputscale.item(), outpuscale: %.3f .base_kernel
                self.model.likelihood.noise.item()))
            if epoch % self.print_freq == 0 or epoch == epochs-1:
                print("Epoch: {}, Avg_loss: {}".format(epoch, avg_loss))
                #plot_posterior_2d(data_loader, self.model, 'training '+str(epoch), self.args)

            self.epoch_loss_history.append(avg_loss)
            if early_stopping is not None:
                if avg_loss < early_stopping:
                    break

    def train_rl_ctx(self, data_loader, epochs, early_stopping=None):

        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                t = time.time()
                # Zero backprop gradients
                self.optimizer.zero_grad()
                # Get output from model
                x, y, num_points = data  # add , num_points
                # divide context (N-1) and target (1)
                x_context, y_context, _, _ = context_target_split(x[:,:num_points,:], y[:,:num_points,:],
                                                                  num_points.item()-1, 1)

                self.model.set_train_data(inputs=x_context, targets=y_context.view(-1), strict=False)
                predictions = self.model(x_context)
                loss = -self.mll(predictions, y_context.view(-1))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                avg_loss = epoch_loss / len(data_loader)
            print('epoch %d - Loss: %.3f   lengthscale: %.9f  outpuscale: %.3f   noise: %.3f' % (epoch, loss.item(),
                self.model.covar_module.base_kernel.base_kernel.lengthscale.item(),
                self.model.covar_module.base_kernel.outputscale.item(),
                self.model.likelihood.noise.item()))
            if epoch % self.print_freq == 0 or epoch == epochs-1:
                print("Epoch: {}, Avg_loss: {}".format(epoch, avg_loss))
                #plot_posterior_2d(data_loader, self.model, 'training '+str(epoch), self.args)

            self.epoch_loss_history.append(avg_loss)
            if early_stopping is not None:
                if avg_loss < early_stopping:
                    break


    def train(self, data_loader, epochs, early_stopping=None):
        print('std')
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                t = time.time()
                # Zero backprop gradients
                self.optimizer.zero_grad()
                # Get output from model
                x, y = data  # add , num_points

                x_context, y_context, x_target, y_target = context_target_split(x[0:1], y[0:1],
                                                                                self.num_context,
                                                                                self.num_target)
                self.model.set_train_data(inputs=x_context, targets=y_context.view(-1), strict=False)
                #self.model.eval()
                #self.model.likelihood.eval()
                #with gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var(False):
                output = self.model(x_context)
                if any(torch.isnan(output.stddev.view(-1))):
                    print('nan at epoch ', epoch)
                    continue
                #self.model.train()
                #self.model.likelihood.train()
                # Calc loss and backprop derivatives
                loss = -self.mll(output, y_context.view(-1))  # .sum()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                avg_loss = epoch_loss / len(data_loader)
            #print('epoch %d - Loss: %.3f   lengthscale: %.3f  outpuscale: %.3f   noise: %.3f' % (epoch, loss.item(),
            #    self.model.covar_module.base_kernel.base_kernel.lengthscale.item(),
            #    self.model.covar_module.base_kernel.outputscale.item(),
            #    self.model.likelihood.noise.item()))
            if epoch % self.print_freq == 0 or epoch == epochs - 1:
                print("Epoch: {}, Avg_loss: {}".format(epoch, avg_loss))
                #plot_posterior(data_loader, self.model, 'training ' + str(epoch), self.args,
                #               title='epoch #{}, avg_los {}'.format(epoch, avg_loss), num_func=1)

            self.epoch_loss_history.append(avg_loss)

            if early_stopping is not None:
                if avg_loss < early_stopping:
                    break


class DKMTrainer_loo():

    def __init__(self, device, model, optimizer, args, print_freq=100):

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        self.print_freq = print_freq
        self.steps = 0
        self.epoch_loss_history = []
        self.args = args

    def train_rl(self, data_loader, epochs, early_stopping=None):
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
            for i, data in enumerate(data_loader):
                t = time.time()
                # Zero backprop gradients
                self.optimizer.zero_grad()
                data = episode_fixed_list[i]
                x, y, num_points = data
                #index = random.randint(0, num_points.item() - 1)
                #x_target = x[:, index, :].unsqueeze(0)
                #y_target = y[:, index, :].view(-1)
                x_target = x.unsqueeze(0)
                y_target = y.view(-1)
                x_context, y_context = merge_context(one_out_list[i])
                #x_context, y_context, _, _ = context_target_split(all_x_context, all_y_context,
                #                                                  num_context=all_x_context.shape[-2]//2,
                #                                                  num_extra_target=0)
                # Get output from model
                self.model.set_train_data(inputs=x_context, targets=y_context.view(-1), strict=False)
                self.model.eval()
                self.model.likelihood.eval()
                with gpytorch.settings.use_toeplitz(False):
                    predictions = self.model(x_target)
                if any(torch.isnan(predictions.stddev.view(-1))):
                    print('found Nan')
                    #self.model.eval()
                    #self.model.likelihood.eval()
                    #predictions = self.model(x_target)

                self.model.train()
                self.model.likelihood.train()

                loss = -self.mll(predictions, y_target.view(-1))

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                avg_loss = epoch_loss / len(data_loader)
            if epoch % self.print_freq == 0 or epoch == epochs-1:
                print("Epoch: {}, Avg_loss: {}".format(epoch, avg_loss))
                #plot_posterior_2d(data_loader, self.model, 'training '+str(epoch), self.args)

            self.epoch_loss_history.append(avg_loss)

            if early_stopping is not None:
                if avg_loss < early_stopping:
                    break

    def train_rl_pick(self, data_loader, epochs, early_stopping=None):
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
            for i, data in enumerate(data_loader):
                t = time.time()
                # Zero backprop gradients
                self.optimizer.zero_grad()

                all_context_points = one_out_list[i]
                data = episode_fixed_list[i]

                x, y, num_points = data
                index = random.randint(0, num_points.item()-1)
                x_target = x[:, index, :].unsqueeze(0)
                y_target = y[:, index, :].view(-1)
                #x_target = x.unsqueeze(0)
                #y_target = y.view(-1)
                x_context, y_context = get_close_context(index, x_target, all_context_points,
                                                        None, num_tot_context=self.args.num_context)
                # Get output from model
                self.model.set_train_data(inputs=x_context, targets=y_context.view(-1), strict=False)
                self.model.eval()
                self.model.likelihood.eval()
                predictions = self.model(x_target)
                if any(torch.isnan(predictions.stddev.view(-1))):
                    print('found Nan')
                    #self.model.eval()
                    #self.model.likelihood.eval()
                    #predictions = self.model(x_target)

                self.model.train()
                self.model.likelihood.train()

                loss = -self.mll(predictions, y_target.view(-1))

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                avg_loss = epoch_loss / len(data_loader)
            if epoch % self.print_freq == 0 or epoch == epochs-1:
                print("Epoch: {}, Avg_loss: {}".format(epoch, avg_loss))
                #plot_posterior_2d(data_loader, self.model, 'training '+str(epoch), self.args)

            self.epoch_loss_history.append(avg_loss)

            if early_stopping is not None:
                if avg_loss < early_stopping:
                    break

