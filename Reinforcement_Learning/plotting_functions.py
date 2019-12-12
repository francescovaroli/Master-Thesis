import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
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

def create_plot_grid(env, args):
    bounds_high = env.observation_space.high
    bounds_low = env.observation_space.low
    if not args.use_running_state:
        x1 = np.linspace(bounds_low[0], bounds_high[0], 100)
        x2 = np.linspace(bounds_low[1], bounds_high[1], 100)
    else:
        x1 = np.linspace(-2, 2, 100)
        x2 = np.linspace(-2, 2, 100)
    X1, X2 = np.meshgrid(x1, x2)

    grid = torch.zeros(100, 2)
    for i in range(2):
        grid_diff = float(bounds_high[i] - bounds_low[i]) / (100 - 2)
        grid[:, i] = torch.linspace(bounds_low[i] - grid_diff, bounds_high[i] + grid_diff, 100)

    x = gpytorch.utils.grid.create_data_from_grid(grid)
    x = x.unsqueeze(0).to(args.dtype).to(args.device_np)
    return x, X1, X2, x1, x2

def plot_NP_policy(policy_np, context_xy, iter_pred, avg_rew, env, args, num_samples=1):
    num_test_context = 999
    policy_np.training = False
    x, X1, X2, x1, x2 = create_plot_grid(env, args)

    # Plot a realization
    Z_distr = policy_np(context_xy[0], context_xy[1], x)  # B x num_points x z_dim  (B=1)
    Z_mean = Z_distr.mean.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
    Z_stddev = Z_distr.stddev.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim

    name = 'NP policy for iteration {}, avg rew {}'.format(iter_pred, int(avg_rew))
    fig = plt.figure(figsize=(16,14)) #figsize=plt.figaspect(1.5)
    fig.suptitle(name, fontsize=20)
    fig.tight_layout()
    # fig.subplots_adjust(top=0.5, wspace=0.3, bottom=0.2)
    ax_mean = fig.add_subplot(221, projection='3d')
    ax_mean.plot_surface(X1, X2, Z_mean.cpu().numpy(), cmap='viridis',  vmin=-1., vmax=1.)
    set_labels(ax_mean)
    set_limits(ax_mean, env, args)

    ax_mean.set_title('Mean of the NP policy', pad=20, fontsize=16)

    ax_stdv = fig.add_subplot(222, projection='3d')
    set_limits(ax_stdv, env, args)
    set_labels(ax_stdv)
    ax_stdv.set_title('Standard deviation of the NP policy', pad=20, fontsize=14)
    stddev_low = Z_mean - Z_stddev
    stddev_high = Z_mean + Z_stddev

    i = 0
    for y_slice in x2:
        ax_stdv.add_collection3d(
            plt.fill_between(x1, stddev_low[i, :].cpu(), stddev_high[i, :].cpu(), color='lightseagreen', alpha=0.2),
            zs=y_slice, zdir='y')
        i += 1

    ax_context = fig.add_subplot(223, projection='3d')
    set_labels(ax_context)
    set_limits(ax_context, env, args)
    ax_context.set_title('Context points improved', pad=20, fontsize=16)
    z = context_xy[1][0,:,0].detach().cpu().numpy()
    xs_context = context_xy[0][0,:,0].detach().cpu().numpy()
    ys_context = context_xy[0][0,:,1].detach().cpu().numpy()
    ax_context.scatter(xs_context, ys_context, z, s=8, c=z, cmap='viridis',  vmin=-1., vmax=1.)


    ax_samples = fig.add_subplot(224, projection='3d')
    ax_samples.set_title(str(num_samples) + ' samples from policy', pad=20, fontsize=16)
    set_limits(ax_samples, env, args)
    set_labels(ax_samples)
    for sample in range(num_samples):
        Z_sample = Z_distr.sample().detach()[0].reshape(X1.shape)
        ax_samples.plot_surface(X1, X2, Z_sample.cpu().numpy(), cmap='viridis', vmin=-1., vmax=1., alpha=0.2)

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
        z = trajectory[1][:,0].detach().cpu().numpy()
        xs_context = trajectory[0][:,0].detach().cpu().numpy()
        ys_context = trajectory[0][:,1].detach().cpu().numpy()
        ax.scatter(xs_context, ys_context, z, c=z, cmap='viridis', alpha=0.1)
    fig.savefig(args.directory_path + '/policy/'+'/Training/'+name)
    plt.close(fig)

def plot_improvements(batch, improved_context, env, i_iter, args):

    episode = batch[-1]
    states = []
    means = []
    actions = []
    disc_rew = []
    num_c = len(episode)
    for transition in episode:
        states.append(transition.state)
        disc_rew.append(transition.disc_rew)
        actions.append(transition.action)
        means.append(transition.mean)
    if args.improve_mean:
        previous = means
    else:
        previous = actions
    name = 'Improvement iter '+ str(i_iter)
    fig = plt.figure(figsize=(16,6))
    fig.suptitle(name, fontsize=20)
    ax = fig.add_subplot(121, projection='3d')
    name = 'Context improvement iter '+ str(i_iter)
    ax.set_title(name)
    set_limits(ax, env, args)
    set_labels(ax)
    start_ep = improved_context[0].shape[1] - num_c
    z = improved_context[1][0,start_ep:,0].detach().cpu().numpy()
    xs_context = improved_context[0][0,start_ep:,0].detach().cpu().numpy()
    ys_context = improved_context[0][0,start_ep:,1].detach().cpu().numpy()
    state_1 = [state[0] for state in states]
    state_2 = [state[1] for state in states]
    ax.scatter(state_1, state_2, previous, c='r', label='sampled',  alpha=0.2)
    ax.scatter(xs_context, ys_context, z, c='y', label='improved', alpha=0.3)
    leg = ax.legend(loc="upper right")
    ax_rew = fig.add_subplot(122, projection='3d')
    set_labels(ax_rew)
    set_limits(ax_rew, env, args)
    a = ax_rew.scatter(state_1, state_2, actions, c=disc_rew, cmap='viridis', alpha=0.5)
    #plt.colorbar()
    cb = fig.colorbar(a)
    cb.set_label('Discounted rewards')
    ax_rew.set_title('Discounted rewards')
    fig.savefig(args.directory_path+'/policy/'+'/Mean improvement/'+name)
    plt.close(fig)

def plot_rewards_history(avg_rewards, args):
    fig_rew, ax_rew = plt.subplots(1, 1)
    ax_rew.plot(np.arange(len(avg_rewards)), avg_rewards)
    ax_rew.set_xlabel('iterations')
    ax_rew.set_ylabel('average reward')
    fig_rew.savefig(args.directory_path + '/average reward')
    plt.close(fig_rew)

def plot_NP_value(value_np, states, disc_rew, values, r_est, env, args, i_iter):
    name = 'Value estimate' + str(i_iter)
    fig = plt.figure(figsize=(16, 14))  # figsize=plt.figaspect(1.5)
    fig.suptitle(name, fontsize=20)
    fig.tight_layout()
    value_np.training = False
    x, X1, X2, x1, x2 = create_plot_grid(env, args)
    # Plot a realization
    V_distr = value_np(states, disc_rew, x)  # B x num_points x z_dim  (B=1)
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
    set_limits(ax_stdv, env, args, np_id='value')
    set_labels(ax_stdv, np_id='value')
    ax.scatter(states[0, :, 0].detach(), states[0, :, 1].detach(), disc_rew[0, :, 0].detach(), c='r', label='Discounted rewards')
    ax.scatter(states[0, :, 0].detach(), states[0, :, 1].detach(), values[0, :, 0].detach(), c='b', label='Estimated values')
    leg = ax.legend(loc="upper right")
    set_labels(ax, np_id='value')
    ax_diff = fig.add_subplot(224, projection='3d')
    set_labels(ax_diff, np_id='value')
    set_limits(ax_stdv, env, args, np_id='value')
    ax_diff.set_title('Difference btw Disc. rew and Values')
    ax_diff.scatter(states[0, :, 0].detach(), states[0, :, 1].detach(), r_est[0, :, 0].detach(), c='g', label='R-V')
    fig.savefig(args.directory_path+'/value/'+name)
    plt.close(fig)