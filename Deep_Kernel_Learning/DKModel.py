import torch
import gpytorch
from utils import context_target_split
import time
from plotting_functions_DKL import plot_posterior

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, x_dim, h_dim, out_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(x_dim, h_dim))  # default h_dim = 1000
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(h_dim, h_dim//2))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(h_dim//2, h_dim//20))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(h_dim//20, out_dim))


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, h_dim, z_dim, name_id):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=z_dim)),
            num_dims=z_dim, grid_size=100)
        self.likelihood = likelihood
        num_points, x_dim = train_x.size()
        self.feature_extractor = LargeFeatureExtractor(x_dim, h_dim, z_dim)
        self.id = name_id

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # We're also scaling the features so that they're nice values
        projected_x = self.feature_extractor(x)
        projected_x = projected_x - projected_x.min(-2)[0]
        projected_x = 2 * (projected_x / projected_x.max(-2)[0]) - 1

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKMTrainer():

    def __init__(self, device, model, optimizer, args, print_freq=100):

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        self.num_context = args.num_context
        self.num_target = args.num_target
        self.print_freq = print_freq
        self.steps = 0
        self.epoch_loss_history = []
        self.args = args

    def train(self, data_loader, epochs, early_stopping=None):
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

                #num_points = min(num_points).item()
                self.model.set_train_data(x_context.squeeze(0), y_context.squeeze(0).squeeze(-1))
                #self.model.train()
                #self.model.likelihood.train()
                #self.model.training = False
                output = self.model(x_context.squeeze(0))
                # Calc loss and backprop derivatives
                loss = -self.mll(output, y_context.squeeze(-1))  #.sum()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                avg_loss = epoch_loss / len(data_loader)
            if epoch % self.print_freq == 0 or epoch == epochs-1:
                print("Epoch: {}, Avg_loss: {}".format(epoch, avg_loss))
                plot_posterior(data_loader, self.model, 'training '+str(epoch), self.args,
                               title='epoch #{}, avg_los {}'.format(epoch, avg_loss), num_func=1)

            self.epoch_loss_history.append(avg_loss)

            if early_stopping is not None:
                if avg_loss < early_stopping:
                    break