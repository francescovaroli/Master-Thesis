import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import randint
from utils_rl import *
import gpytorch

def set_labels(ax, np_id='policy'):
    ax.set_xlabel('Position', fontsize=14)
    ax.set_ylabel('Velocity', fontsize=14)
    if np_id == 'value':
        ax.set_zlabel('Reward', fontsize=14)
    else:
        ax.set_zlabel('Acceleration', fontsize=14)

def set_limits(ax, env, args, np_id='policy'):
    bounds_high = env.observation_space.high
    bounds_low = env.observation_space.low
    if np_id == 'value':
        ax.set_xlim(bounds_low[0], bounds_high[0])
        ax.set_ylim(bounds_low[1], bounds_high[1])
        return
    if args.use_running_state:
        ax.set_zlim(env.action_space.low, env.action_space.high)
        return

    ax.set_xlim(bounds_low[0], bounds_high[0])
    ax.set_ylim(bounds_low[1], bounds_high[1])
    ax.set_zlim(env.action_space.low, env.action_space.high)

def create_plot_grid(env, args, size=20):
    bounds_high = env.observation_space.high
    bounds_low = env.observation_space.low
    if not args.use_running_state:
        x1 = np.linspace(bounds_low[0], bounds_high[0], size)
        x2 = np.linspace(bounds_low[1], bounds_high[1], size)
    else:
        x1 = np.linspace(-2, 2, size)
        x2 = np.linspace(-2, 2, size)
    X1, X2 = np.meshgrid(x1, x2)

    grid = torch.zeros(size, 2)
    for i in range(2):
        grid_diff = float(bounds_high[i] - bounds_low[i]) / (size - 2)
        grid[:, i] = torch.linspace(bounds_low[i] - grid_diff, bounds_high[i] + grid_diff, size)

    x = gpytorch.utils.grid.create_data_from_grid(grid)
    x = x.unsqueeze(0).to(args.dtype).to(args.device_np)
    return x, X1, X2, x1, x2

def plot_NP_policy(policy_np, all_context_xy, iter_pred, avg_rew, env, args, colors):
    num_test_context = 999
    policy_np.training = False
    x, X1, X2, x1, x2 = create_plot_grid(env, args)

    # set up axes
    name = 'NP policy for iteration {}, avg rew {}'.format(iter_pred, int(avg_rew))
    fig = plt.figure(figsize=(16, 14))  # figsize=plt.figaspect(1.5)
    fig.suptitle(name, fontsize=20)
    fig.tight_layout()
    ax_mean = fig.add_subplot(221, projection='3d')
    set_labels(ax_mean)
    set_limits(ax_mean, env, args)
    ax_mean.set_title('Mean of the NP policies', pad=20, fontsize=16)

    ax_context = fig.add_subplot(223, projection='3d')
    set_labels(ax_context)
    set_limits(ax_context, env, args)
    ax_context.set_title('Context points improved', pad=20, fontsize=16)

    ax_samples = fig.add_subplot(224, projection='3d')
    ax_samples.set_title(' samples from policy', pad=20, fontsize=16)
    set_limits(ax_samples, env, args)
    set_labels(ax_samples)
    # Plot a realization
    for e, context_xy in enumerate(all_context_xy):
        context_x, context_y, real_len = context_xy
        Z_distr = policy_np(context_x[:real_len], context_y[:real_len], x)  # B x num_points x z_dim  (B=1)
        Z_mean = Z_distr.mean.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
        Z_stddev = Z_distr.stddev.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim

        ax_mean.plot_surface(X1, X2, Z_mean.cpu().numpy(), cmap='viridis',  vmin=-1., vmax=1.)

        stddev_low = Z_mean - Z_stddev
        stddev_high = Z_mean + Z_stddev
        if e == 0:
            ax_stdv = fig.add_subplot(222, projection='3d')
            set_limits(ax_stdv, env, args)
            set_labels(ax_stdv)
            ax_stdv.set_title('Standard deviation of one NP policy', pad=20, fontsize=14)
            i = 0
            for y_slice in x2:
                ax_stdv.add_collection3d(
                    plt.fill_between(x1, stddev_low[i, :].cpu(), stddev_high[i, :].cpu(), color='lightseagreen',
                                     alpha=0.2),
                    zs=y_slice, zdir='y')
                i += 1

        ax_context.scatter(context_x[0, :real_len, 0], context_x[0, :real_len, 1], context_y[0, :real_len, 0],
                           s=8, c=colors[e])

        Z_sample = Z_distr.sample().detach()[0].reshape(X1.shape)
        ax_samples.plot_surface(X1, X2, Z_sample.cpu().numpy(), color=colors[e], alpha=0.2)

    # plt.show()
    fig.savefig(args.directory_path + '/policy/'+'/NP estimate/'+name, dpi=250)
    plt.close(fig)


def plot_training_set(i_iter, replay_memory, env, args):
    name = 'Training trajectories ' + str(i_iter)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_title(name)
    set_limits(ax, env, args)
    set_labels(ax)
    for trajectory in replay_memory:
        real_len = trajectory[2]
        z = trajectory[1][:real_len,0].detach().cpu().numpy()
        xs_context = trajectory[0][:real_len,0].detach().cpu().numpy()
        ys_context = trajectory[0][:real_len,1].detach().cpu().numpy()
        ax.scatter(xs_context, ys_context, z, c=z, cmap='viridis', alpha=0.1, vmin=-1., vmax=1.)
    fig.savefig(args.directory_path + '/policy/'+'/Training/'+name)
    plt.close(fig)

def plot_improvements(all_dataset, est_rewards, env, i_iter, args, colors):

    name = 'Improvement iter ' + str(i_iter)
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(name, fontsize=20)
    ax = fig.add_subplot(121, projection='3d')
    name_c = 'Context improvement iter ' + str(i_iter)
    ax.set_title(name_c)
    set_limits(ax, env, args)
    set_labels(ax)
    ax_rew = fig.add_subplot(122, projection='3d')
    set_labels(ax_rew)
    set_limits(ax_rew, env, args)
    for e, episode in enumerate(all_dataset):
        real_len = episode['real_len']
        states = episode['states'][:real_len]
        disc_rew = episode['discounted_rewards'][:real_len]
        actions = episode['actions'][:real_len]
        means = episode['means'][:real_len]
        new_means = episode['new_means'][:real_len]
        est_rew = est_rewards[e]
        if e == 0:
            ax.scatter(states[:, 0].numpy(), states[:, 1].numpy(), means[:, 0].numpy(), c=colors[e], s=2, label='sampled', alpha=0.1)
            ax.scatter(states[:, 0].numpy(), states[:, 1].numpy(), new_means[:, 0].numpy(), c=colors[e], marker='+', label='improved', alpha=0.5)
        else:
            ax.scatter(states[:, 0].numpy(), states[:, 1].numpy(), means[:, 0].numpy(), c=colors[e], s=2, alpha=0.1)
            ax.scatter(states[:, 0].numpy(), states[:, 1].numpy(), new_means[:, 0].numpy(), c=colors[e], marker='+', alpha=0.5)

        a = ax_rew.scatter(states[:, 0].numpy(), states[:, 1].numpy(), actions[:, 0].numpy(), c=est_rew[:], cmap='viridis', alpha=0.5)

    leg = ax.legend(loc="upper right")
    cb = fig.colorbar(a)
    cb.set_label('Discounted rewards')
    ax_rew.set_title('Discounted rewards')
    fig.savefig(args.directory_path+'/policy/Mean improvement/'+name)
    plt.close(fig)


def plot_rewards_history(avg_rewards, args):
    fig_rew, ax_rew = plt.subplots(1, 1)
    ax_rew.plot(np.arange(len(avg_rewards)), avg_rewards)
    ax_rew.set_xlabel('iterations')
    ax_rew.set_ylabel('average reward')
    fig_rew.savefig(args.directory_path + '/average reward')
    plt.close(fig_rew)

def plot_NP_value(value_np, all_states, all_values, all_episodes, all_rews,env, args, i_iter):
    name = 'Value estimate' + str(i_iter)
    fig = plt.figure(figsize=(16, 14))  # figsize=plt.figaspect(1.5)
    fig.suptitle(name, fontsize=20)
    fig.tight_layout()
    value_np.training = False
    x, X1, X2, x1, x2 = create_plot_grid(env, args)
    # Plot a realization
    V_distr = value_np(all_states, all_rews, x)  # B x num_points x z_dim  (B=1)
    V_mean = V_distr.mean.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
    V_stddev = V_distr.stddev.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
    stddev_low = V_mean - V_stddev
    stddev_high = V_mean + V_stddev
    vmin = stddev_low.min()
    vmax = stddev_high.max()
    ax_mean = fig.add_subplot(221, projection='3d')
    ax_mean.plot_surface(X1, X2, V_mean.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
    set_labels(ax_mean)
    set_limits(ax_mean, env, args, np_id='value')
    ax_mean.set_zlim(vmin, vmax)

    ax_mean.set_title('Mean of the value NP', pad=20, fontsize=16)

    ax_stdv = fig.add_subplot(222, projection='3d')
    set_limits(ax_stdv, env, args, np_id='value')
    set_labels(ax_stdv, np_id='value')
    ax_stdv.set_title('Standard deviation of the value NP', pad=20, fontsize=14)


    ax_stdv.plot_surface(X1, X2, stddev_low.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
    ax_stdv.plot_surface(X1, X2, stddev_high.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
    i = 0
    for y_slice in x2:
        ax_stdv.add_collection3d(
            plt.fill_between(x1, stddev_low[i, :].cpu(), stddev_high[i, :].cpu(), color='lightseagreen', alpha=0.2),
            zs=y_slice, zdir='y')
        i += 1

    ax = fig.add_subplot(223, projection='3d')
    set_limits(ax, env, args, np_id='value')
    set_labels(ax, np_id='value')
    leg = ax.legend(loc="upper right")
    ax_diff = fig.add_subplot(224, projection='3d')
    set_labels(ax_diff, np_id='value')
    set_limits(ax_diff, env, args, np_id='value')
    for episode, values in zip(all_episodes, all_values):
        real_len = episode['real_len']
        states = episode['states'][:real_len].unsqueeze(0)
        disc_rew = episode['discounted_rewards'][:real_len].unsqueeze(0)
        r_est = disc_rew - values
        ax.scatter(states[0, :, 0].detach(), states[0, :, 1].detach(), disc_rew[0, :, 0].detach(), c='r', label='Discounted rewards',alpha=0.1)
        ax.scatter(states[0, :, 0].detach(), states[0, :, 1].detach(), values[0, :, 0].detach(), c='b', label='Estimated values',alpha=0.1)
    ax_diff.scatter(states[0, :, 0].detach(), states[0, :, 1].detach(), r_est[0, :, 0].detach(), c='g', label='R-V')
    ax_diff.set_title('Difference btw one disc. rew and Values')
    fig.savefig(args.directory_path+'/value/'+name)
    plt.close(fig)


def plot_initial_context(context_points, colors, env, args, i_iter):
    name = 'Contexts of all episodes at iter ' + str(i_iter)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.set_title(name)
    set_limits(ax, env, args)
    set_labels(ax)
    for e, episode in enumerate(context_points):
        real_len = episode[2]
        z = episode[1][:, :real_len, 0].detach().cpu().numpy()
        xs_context = episode[0][:, :real_len, 0].detach().cpu().numpy()
        ys_context = episode[0][:, :real_len, 1].detach().cpu().numpy()
        ax.scatter(xs_context, ys_context, z, c=colors[e], alpha=0.5)
    fig.savefig(args.directory_path + '/policy/'+ '/All policies samples/' + name)
    plt.close(fig)