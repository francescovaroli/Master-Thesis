import pickle
import matplotlib.pyplot as plt

file = '/home/francesco/PycharmProjects/MasterThesis/plots/NP&ANP/2D/'
name = 'learned_models_ANP2e_4b_(190, 250)c(200, 350)t_256x128x256.p'

model = pickle.load(file+name)


def plot_posterior():
    for batch in data_loader:
        break

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
    ax_real.plot_surface(X1, X2, y.reshape(X1.shape).cpu().numpy(), cmap='viridis', vmin=-1., vmax=1.)
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
    sigma = sigma.numpy()
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
                             alpha=0.1),
            zs=y_slice, zdir='y')
        i += 1
    # Extract mean of distribution
    ax_mean.plot_surface(X1, X2, mu, cmap='viridis', vmin=-1., vmax=1.)
    for ax in [ax_mean, ax_context, ax_real]:
        ax.set_zlim(min_mu, max_mu)
    ax_mean.set_title('Posterior estimate')
    plt.savefig(args.directory_path + '/posteriior' + id, dpi=250)
    #plt.show()
    plt.close(fig)
    return
