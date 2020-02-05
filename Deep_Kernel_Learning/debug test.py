import gpytorch
import torch
import numpy as np

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
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=z_dim)), num_dims=z_dim, grid_size=grid_size)
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
            z = projected_x / projected_x.max(-2)[0]

        elif self.scaling == 'normal':
            z = (projected_x - projected_x.mean()) / projected_x.std()

        elif self.scaling is None:
            z = projected_x

        z[torch.isnan(z)] = 10000
        return z

    def forward(self, x):

        z = self.project(x)
        mean_x = self.mean_module(z)
        covar_x = self.covar_module(z)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def context_target_split(x, y, num_context, num_extra_target):

    num_points = x.shape[1]
    # Sample locations of context and target points
    locations = np.random.choice(num_points,
                                 size=num_context + num_extra_target,
                                 replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations[num_context:], :]
    y_target = y[:, locations[num_context:], :]
    return x_context, y_context, x_target, y_target

epochs = 30
x_init = torch.randn([1, 1000, 4])
y_init = torch.randn([1, 1000, 1])

likelihood = gpytorch.likelihoods.GaussianLikelihood()  # noise_constraint=gpytorch.constraints.LessThan(args.noise)
model = GPRegressionModel(x_init, y_init.view(-1), likelihood,
                          h_dim=200, z_dim=1, name_id='DKL', scaling='uniform')
mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)


optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()}], lr=0.01)

## train
for epoch in range(epochs):
    epoch_loss = 0.
    for i, data in enumerate(data_loader):

        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        x, y, num_points = data  # add , num_points
        # divide context (N-1) and target (1)
        x_context, y_context, x_target, y_target = context_target_split(x[:, :num_points, :],
                                                                        y[:, :num_points, :],
                                                                        num_points.item() - 1, 1)

        # num_points = min(num_points).item()
        model.set_train_data(inputs=x_context, targets=y_context.view(-1), strict=False)
        model.eval()
        model.likelihood.eval()
        predictions = model(x_target)
        model.train()
        model.likelihood.train()

        loss = -mll(predictions, y_target.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        avg_loss = epoch_loss / len(data_loader)
    if epoch % 10 == 0 or epoch == epochs - 1:
        print("Epoch: {}, Avg_loss: {}".format(epoch, avg_loss))

