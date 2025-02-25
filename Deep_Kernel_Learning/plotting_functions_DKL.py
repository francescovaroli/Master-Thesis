import matplotlib.pyplot as plt
import torch
import gpytorch
from utils.utils import context_target_split
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_posterior(data_loader, model, id, args, title='Posterior', num_func=4):
    plt.ylabel('Predicted y distribution')
    colors = ['r', 'b', 'g', 'y', 'b', 'g', 'y']
    for j in range(num_func):
        plt.figure(j)
        plt.xlabel('x')

        x, y = data_loader.dataset.data[j]
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        x_context, y_context, x_target, y_target = context_target_split(x[0:1], y[0:1],
                                                                        args.num_context,
                                                                        args.num_target)

        #plt.title(title)
        model.set_train_data(x_context.squeeze(0), y_context.squeeze(0).squeeze(-1), strict=False)
        model.training = False
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            p_y_pred = model(x[0:1].squeeze(0))
        model.training = True
        # Extract mean of distribution
        mu = p_y_pred.loc.detach().cpu().numpy()
        stdv = p_y_pred.stddev.detach().cpu().numpy()

        plt.plot(x[0:1].cpu().numpy()[0].squeeze(-1), mu,
                 alpha=0.9, c=colors[j], label='Mean')
        plt.fill_between(x[0:1].cpu().numpy()[0].squeeze(-1), mu - stdv, mu + stdv, color=colors[j], alpha=0.1, label='stdev')

        plt.plot(x[0].cpu().numpy(), y[0].cpu().numpy(), alpha=0.5, c='k', label='Real function')
        plt.scatter(x_context[0].cpu().numpy(), y_context[0].cpu().numpy(), c=colors[j], label='Context points')
        plt.legend()
        plt.savefig(args.directory_path + '/'+id+str(j))
    plt.close()


def create_plot_grid(extent, args, size=20):

    x1 = np.linspace(extent[0], extent[1], size)
    x2 = np.linspace(extent[2], extent[3], size)
    X1, X2 = np.meshgrid(x1, x2)

    grid = torch.zeros(size, 2)
    for i in range(2):
        grid_diff = float(extent[i*2] - extent[2*i+1]) / (size - 2)
        grid[:, i] = torch.linspace(extent[i*2] - grid_diff, extent[2*i+1] + grid_diff, size)

    x = gpytorch.utils.grid.create_data_from_grid(grid)
    x = x.unsqueeze(0)
    return x, X1, X2, x1, x2


def plot_posterior_2d(data_loader, model, id, args):
    for n, batch in enumerate(data_loader):

        # Use batch to create random set of context points
        x, y = batch  # , real_len

        x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1],
                                                          args.num_context,
                                                          args.num_target)
        x, X1, X2, x1, x2 = create_plot_grid(args.extent, args, size=args.grid_size)

        fig = plt.figure(figsize=(20, 8))  # figsize=plt.figaspect(1.5)
        fig.suptitle(id, fontsize=20)
        #fig.tight_layout()

        ax_real = fig.add_subplot(131, projection='3d')
        ax_real.plot_surface(X1, X2, y.reshape(X1.shape).cpu().numpy(), cmap='viridis')
        ax_real.set_title('Real function')

        ax_context = fig.add_subplot(132, projection='3d')
        ax_context.scatter(x_context[0,:,0].detach().cpu().numpy(),
                           x_context[0, :, 1].detach().cpu().numpy(),
                           y_context[0,:,0].detach().cpu().numpy(),
                           c=y_context[0,:,0].detach().cpu().numpy(),
                           cmap='viridis', vmin=-1., vmax=1.,  s=8)

        ax_context.set_title('Context points')
        model.set_train_data(x_context.squeeze(0), y_context.squeeze(0).squeeze(-1), strict=False)
        model.training = False
        with torch.no_grad():
            p_y_pred = model(x[0:1])
        mu = p_y_pred.mean.reshape(X1.shape).cpu()
        mu[torch.isnan(mu)] = 0.
        mu = mu.numpy()
        sigma = p_y_pred.stddev.reshape(X1.shape).cpu()
        sigma[torch.isnan(sigma)] = 0.
        sigma = sigma.detach().numpy()
        std_h = mu + sigma
        std_l = mu - sigma
        model.training = True
        max_mu = std_h.max()
        min_mu = std_l.min()
        ax_mean = fig.add_subplot(133, projection='3d')
        i = 0
        for y_slice in x2:
            ax_mean.add_collection3d(
                plt.fill_between(x1, std_l[i, :], std_h[i, :], color='lightseagreen',
                                 alpha=0.04, label='stdev'),
                zs=y_slice, zdir='y')
            i += 1
        # Extract mean of distribution
        ax_mean.plot_surface(X1, X2, mu, cmap='viridis', label='mean')
        for ax in [ax_mean, ax_context, ax_real]:
            ax.set_zlim(min_mu, max_mu)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('y')
        ax_mean.set_title('Posterior estimate')
        ax_mean.set_xlim(args.extent[0], args.extent[1])
        ax_mean.set_ylim(args.extent[2], args.extent[3])
        plt.savefig(args.directory_path + '/posteriior' + id+str(n), dpi=250)
        #plt.show()
        plt.close(fig)
    return
