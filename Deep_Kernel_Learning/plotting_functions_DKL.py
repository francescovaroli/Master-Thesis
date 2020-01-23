import matplotlib.pyplot as plt
import torch
import gpytorch
from utils import context_target_split


def plot_posterior(data_loader, model, id, args, title='Posterior', num_func=2):
    plt.figure(4)
    colors = ['r', 'b', 'g', 'y']
    for j in range(num_func):
        for data_test in data_loader:
            break
        x, y = data_test

        x_context, y_context, x_target, y_target = context_target_split(x[0:1], y[0:1],
                                                                        args.num_context,
                                                                        args.num_target)

        plt.title(title)
        model.set_train_data(x_context.squeeze(0), y_context.squeeze(0).squeeze(-1))
        model.training = False
        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            p_y_pred = model(x[0:1].squeeze(0))
        # Extract mean of distribution
        mu = p_y_pred.loc.detach().cpu().numpy()
        stdv = p_y_pred.stddev.detach().cpu().numpy()
        plt.xlabel('x')
        plt.ylabel('means of y distribution')
        plt.plot(x[0:1].cpu().numpy()[0].squeeze(-1), mu,
                 alpha=0.9, c=colors[j])
        plt.fill_between(x[0:1].cpu().numpy()[0].squeeze(-1), mu - stdv, mu + stdv, color=colors[j], alpha=0.1)

        plt.plot(x[0].cpu().numpy(), y[0].cpu().numpy(), alpha=0.5, c='k')
        plt.scatter(x_context[0].cpu().numpy(), y_context[0].cpu().numpy(), c=colors[j], label='context')
    plt.legend()
    plt.savefig(args.directory_path + '/'+id)
    plt.close()
