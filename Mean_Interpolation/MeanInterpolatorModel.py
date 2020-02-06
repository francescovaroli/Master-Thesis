import torch
import numpy as np
from utils import context_target_split, context_target_split_CinT
from utils_rl import *
import matplotlib.pyplot as plt


class Interpolator(torch.nn.Module):
    def __init__(self, input_dim):
        super(Interpolator, self).__init__()
        self.W = torch.nn.Parameter(data=torch.Tensor(input_dim, input_dim), requires_grad=True)
        self.W.data.uniform_(-1, 1)
        self.z_dim = input_dim

    def forward(self, z_context, y_context, z_target):
        z_diff = z_target-z_context
        thetas = torch.exp(-(torch.matmul(z_diff, self.W) * z_diff).sum(-1))
        return y_context.t().matmul(thetas)/thetas.sum()

class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, x_dim, h_dim, out_dim):
        super(FeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(x_dim, h_dim))  # default h_dim = 256
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(h_dim, out_dim))


class MeanInterpolator(torch.nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, scaling=None):
        super(MeanInterpolator, self).__init__()
        self.feature_extractor = FeatureExtractor(x_dim, h_dim, z_dim)
        self.interpolator = Interpolator(z_dim)
        self.scaling = scaling
        self.id = 'MI'

    def project(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # We're also scaling the features so that they're nice values
        projected_x = self.feature_extractor(x.squeeze(0))

        if self.scaling == 'uniform':
            projected_x = projected_x - projected_x.min(-2)[0]
            z = (projected_x / projected_x.max(-2)[0])

        elif self.scaling == 'normal':
            z = (projected_x - projected_x.mean())/projected_x.std()

        elif self.scaling is None:
            z = projected_x

        z[torch.isnan(z)] = 10000
        return z

    def forward(self, x_context, y_context, x_target):
        _, num_context, x_dim = x_context.size()
        if self.scaling is not None:
            encoder_input = torch.cat([x_context, x_target], dim=-2)
            z_full = self.project(encoder_input)
            z_context = z_full[..., :num_context, :]
            z_target = z_full[..., num_context:, :]
        else:
            z_context = self.feature_extractor(x_context.squeeze(0))
            z_target = self.feature_extractor(x_target.squeeze(0))

        return self.interpolator(z_context, y_context.squeeze(0), z_target)


class MITrainer():

    def __init__(self, device, model, optimizer, args, print_freq=100):

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.print_freq = print_freq
        self.steps = 0
        self.epoch_loss_history = []
        self.args = args

    def train_rl(self, data_loader, epochs, early_stopping=None):
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                # Zero backprop gradients
                self.optimizer.zero_grad()
                # Get output from model
                x, y, num_points = data  # add , num_points
                # divide context (N-1) and target (1)
                x_context, y_context, x_target, y_target = context_target_split(x[:,:num_points,:], y[:,:num_points,:],
                                                                  num_points.item()-1, 1)
                prediction = self.model(x_context, y_context, x_target)
                if torch.isnan(prediction):
                    prediction = self.model(x_context, y_context, x_target)
                loss = self._loss(y_target.squeeze(0), prediction.squeeze(0))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                avg_loss = epoch_loss / len(data_loader)

            if epoch % self.print_freq == 0 or epoch == epochs-1 :
                print("Epoch: {}, Avg_loss: {}, W_sum: {}".format(epoch, avg_loss, self.model.interpolator.W.sum().item()))

            self.epoch_loss_history.append(avg_loss)
            if early_stopping is not None:
                if avg_loss < early_stopping:
                    break

    def train(self, data_loader, epochs, early_stopping=None):
        for epoch in range(epochs):
            epoch_loss = 0.
            for i, data in enumerate(data_loader):
                # Zero backprop gradients
                self.optimizer.zero_grad()
                # Get output from model
                x, y = data  # add , num_points
                # divide context (N-1) and target (1)
                x_context, y_context, x_target, y_target = context_target_split(x, y,
                                                                                self.args.num_context,
                                                                                self.args.num_target)
                prediction = self.model(x_context, y_context, x_target)
                loss = self._loss(y_target.squeeze(0), prediction.squeeze(0))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                avg_loss = epoch_loss / len(data_loader)

            if epoch % self.print_freq == 0 or epoch == epochs-1:
                print("Epoch: {}, Avg_loss: {}, W_sum: {}".format(epoch, avg_loss, self.model.interpolator.W.sum().item()))
                plot_z(self.model, self.args, epoch)
            self.epoch_loss_history.append(avg_loss)
            if early_stopping is not None:
                if avg_loss < early_stopping:
                    break

    def _loss(self, y_target, y_pred):
        diff = y_target - y_pred
        return diff.matmul(diff.t())

#test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    z_dim = 2
    inter = Interpolator(z_dim)
    res = []
    inter.W = torch.ones(z_dim, z_dim)/10
    z = torch.randn(15, z_dim)*10
    z_t = z[0,:] #torch.zeros(1,z_dim)
    y_c = torch.randn(15,2)

    y_pred = inter(z, y_c, z_t)

    if z_dim == 1:
        plt.scatter(z, y_c, s=40)
        plt.scatter(z_t, y_pred, c='r', s=98, marker='+')
    else:
        plt.scatter(z[:, 0], z[:, 1], cmap='viridis', c=y_c[:,0], vmin=y_c.min(), vmax=y_c.max())
        plt.scatter(z_t[:1], z_t[1:], s=8, cmap='viridis', c=y_pred, vmin=y_c.min(), vmax=y_c.max())
    plt.show()

def plot_z(model, args, iter_pred):
    colors = ['b', 'r', 'g', 'y', 'k', 'o']
    fig_z, az = plt.subplots(1, 1, figsize=(10,8))
    start, end = args.x_range
    x = torch.linspace(start, end, 100).unsqueeze(1)
    with torch.no_grad():
        z_proj = model.project(x)
    for e, z_dim in enumerate(z_proj.t()):
        az.plot(x.cpu(), z_dim.cpu(), alpha=0.5, color=colors[e], label='z dimension: {}'.format(e))
    plt.legend()
    az.set_title('Z projection of the state space')
    az.set_xlabel('z')
    fig_z.savefig(args.directory_path +'/'+str(iter_pred))
    plt.close(fig_z)