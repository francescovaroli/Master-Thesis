import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import randint
from utils_rl import *
import gpytorch
from torch.distributions import Normal


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


def plot_NP_policy_MC(policy_np, rm, iter_pred, env, args):
    size = 10
    fig = plt.figure(figsize=(16,8))
    #fig.suptitle('NP policy for iteration {}, , avg rew {} '.format(iter_pred, int(avg_rew)), fontsize=20)
    x, X1, X2, xs = create_plot_grid(env, args, size=size)
    with torch.no_grad():
        context_x, context_y = merge_context(rm.data)
        z_distr = policy_np(context_x, context_y, x)  # B x num_points x z_dim  (B=1)
        z_mean = z_distr.mean.detach()[0].reshape(X1.shape)# x1_dim x x2_dim
        z_stddev = z_distr.stddev.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
        stddev_low = z_mean - z_stddev
        stddev_high = z_mean + z_stddev
    ax = fig.add_subplot(1,2,1, projection='3d')
    xp1, xp2 = np.meshgrid(xs[0], xs[2])
    middle_vel = len(X2) // 2

    i = 0
    for y_slice in xs[2]:
        ax.add_collection3d(
            plt.fill_between(xs[0], stddev_low[i, middle_vel, :, middle_vel].cpu(), stddev_high[i, middle_vel, :, middle_vel].cpu(), color='lightseagreen',
                             alpha=0.1),
            zs=y_slice, zdir='y')
        i += 1
    #ax.set_title('cart v: {:.2f}, bar v:{:.2f}'.format(xs[1][middle_vel], xs[3][middle_vel]))
    ax.set_xlabel('cart position')
    ax.set_ylabel('bar angle')
    ax.set_zlabel('action')
    ax.set_zlim(-1, 1)
    ax.set_ylim(xs[2][1], xs[2][-1])
    ax.plot_surface(xp1, xp2, z_mean[:, middle_vel, :, middle_vel].cpu().numpy(), cmap='viridis', vmin=-1., vmax=1.)
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    #ax2.set_title('cart p: {:.2f}, bar angle:{:.2f}'.format(xs[0][middle_vel], xs[2][middle_vel]))
    ax2.set_xlabel('cart velocity')
    ax2.set_ylabel('bar velocity')
    ax2.set_zlabel('action')
    ax2.set_zlim(-1, 1)
    xp1, xp2 = np.meshgrid(xs[1], xs[3])
    ax2.scatter(context_x[..., 0], context_x[..., 1], context_y[..., 0], c=context_y[..., 0], cmap='viridis', vmin=-1., vmax=1.)

    fig.savefig(args.directory_path + '/policy/'+str(iter_pred), dpi=250)
    plt.close(fig)

def plot_NP_policy(policy_np, all_context_xy, rm, iter_pred, avg_rew, env, args, colors):
    num_test_context = 999
    policy_np.training = False
    x, X1, X2, x1, x2 = create_plot_grid(env, args)

    # set up axes
    name = 'NP policy for iteration {}, avg rew {}'.format(iter_pred, int(avg_rew))
    fig = plt.figure(figsize=(16, 14))  # figsize=plt.figaspect(1.5)
    fig.suptitle(name, fontsize=20)
    fig.tight_layout()

    ax_context = fig.add_subplot(222, projection='3d')
    set_labels(ax_context)
    set_limits(ax_context, env, args)
    ax_context.set_title('Context points improved', pad=30, fontsize=16)

    ax_samples = fig.add_subplot(223, projection='3d')
    if args.fixed_sigma is not None:
        ax_samples.set_title('Samples from policy, fixed sigma {:.2f}'.format(args.fixed_sigma.item()), pad=30, fontsize=16)
    else:
        ax_samples.set_title('Samples from policy, sigma from NP', pad=30, fontsize=16)
    set_limits(ax_samples, env, args)
    set_labels(ax_samples)

    stddev_low_list = []
    stddev_high_list = []
    z_distr_list = []
    for e, context_xy in enumerate(all_context_xy):
        with torch.no_grad():
            context_x, context_y, real_len = context_xy
            z_distr = policy_np(context_x[:real_len], context_y[:real_len], x)  # B x num_points x z_dim  (B=1)
            z_distr_list.append(z_distr)
            z_mean = z_distr.mean.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
            z_stddev = z_distr.stddev.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
            stddev_low_list.append(z_mean - z_stddev)
            stddev_high_list.append(z_mean + z_stddev)

        ax_context.scatter(context_x[0, :real_len, 0].cpu(), context_x[0, :real_len, 1].cpu(),
                           context_y[0, :real_len, 0].cpu(), s=8, c=colors[e])

    ax_mean = fig.add_subplot(221, projection='3d')
    for stddev_low, stddev_high in zip(stddev_low_list, stddev_high_list):
        i = 0
        for y_slice in x2:
            ax_mean.add_collection3d(
                plt.fill_between(x1, stddev_low[i, :].cpu(), stddev_high[i, :].cpu(), color='lightseagreen',
                                 alpha=0.1),
                zs=y_slice, zdir='y')
            i += 1
    for e, z_distr in enumerate(z_distr_list):
        z_mean = z_distr.mean.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
        ax_mean.plot_surface(X1, X2, z_mean.cpu().numpy(), cmap='viridis', vmin=-1., vmax=1.)
        if args.fixed_sigma is None:
            sigmas = z_distr.stddev
        else:
            sigmas = args.fixed_sigma * torch.ones_like(z_distr.mean)
        a_distr = Normal(z_distr.mean, sigmas)
        z_sample = a_distr.sample().detach()[0].reshape(X1.shape)
        ax_samples.plot_surface(X1, X2, z_sample.cpu().numpy(), color=colors[e], alpha=0.2)

    ax_rm = fig.add_subplot(224, projection='3d')
    set_limits(ax_rm, env, args)
    set_labels(ax_rm)
    ax_rm.set_title('Training set (RM)', pad=30, fontsize=16)
    for traj in rm:
        r_len = traj[-1]
        ax_rm.scatter(traj[0][:r_len, 0].cpu(), traj[0][:r_len, 1].cpu(), traj[1][:r_len, 0].cpu(),
                      c=traj[1][:r_len, 0].cpu(), alpha=0.1, cmap='viridis')
    set_labels(ax_mean)
    set_limits(ax_mean, env, args)
    ax_mean.set_title('Mean of the NP policies', pad=30, fontsize=16)
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
            ax.scatter(states[:, 0].cpu(), states[:, 1].cpu(), means[:, 0].cpu(), c='k', s=4, label='sampled', alpha=0.3)
            ax.scatter(states[:, 0].cpu(), states[:, 1].cpu(), new_means[:, 0].cpu(), c=new_means[:, 0].cpu(), marker='+', label='improved', alpha=0.3)
            leg = ax.legend(loc="upper right")
        else:
            ax.scatter(states[:, 0].cpu(), states[:, 1].cpu(), means[:, 0].cpu(), c='k', s=4, alpha=0.3)
            ax.scatter(states[:, 0].cpu(), states[:, 1].cpu(), new_means[:, 0].cpu(), c=new_means[:, 0].cpu(), marker='+', alpha=0.3)

        a = ax_rew.scatter(states[:, 0].cpu(), states[:, 1].cpu(), actions[:, 0].cpu(), c=est_rew[:], cmap='viridis', alpha=0.5)

    cb = fig.colorbar(a)
    cb.set_label('Discounted rewards')
    ax_rew.set_title('Discounted rewards')
    fig.savefig(args.directory_path+'/policy/Mean improvement/'+name, dpi=250)
    plt.close(fig)


def plot_rewards_history(avg_rewards, tot_steps, args):
    fig_rew, ax_rew = plt.subplots(1, 1)
    ax_rew.plot(tot_steps, avg_rewards)
    ax_rew.set_xlabel('number of steps')
    ax_rew.set_ylabel('average reward')
    ax_rew.set_title('Average Reward History')
    plt.grid()
    fig_rew.savefig(args.directory_path + '/00_average reward')
    plt.close(fig_rew)

def plot_NP_value(value_np, all_states, target_states, values, all_rews, advantages, env, args, i_iter):
    name = 'Value estimate ' + str(i_iter)
    fig = plt.figure(figsize=(18, 6))  # figsize=plt.figaspect(1.5)
    fig.suptitle(name, fontsize=20)
    fig.tight_layout()
    value_np.training = False
    x, X1, X2, x1, x2 = create_plot_grid(env, args)
    # Plot a realization
    with torch.no_grad():
        V_distr = value_np(all_states, all_rews, x)  # B x num_points x z_dim  (B=1)
        V_mean = V_distr.mean.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
        V_stddev = V_distr.stddev.detach()[0].reshape(X1.shape)  # x1_dim x x2_dim
    stddev_low = V_mean - V_stddev
    stddev_high = V_mean + V_stddev
    vmin = stddev_low.min().cpu()
    vmax = stddev_high.max().cpu()
    ax_mean = fig.add_subplot(131, projection='3d')
    i = 0
    for y_slice in x2:
        ax_mean.add_collection3d(
            plt.fill_between(x1, stddev_low[i, :].cpu(), stddev_high[i, :].cpu(), color='lightseagreen', alpha=0.05),
            zs=y_slice, zdir='y')
        i += 1
    ax_mean.plot_surface(X1, X2, V_mean.cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)
    set_labels(ax_mean, np_id='value')
    set_limits(ax_mean, env, args, np_id='value')
    ax_mean.set_zlim(vmin, vmax)

    ax_mean.set_title('Mean and std. dev. of the value NP', pad=25, fontsize=16)

    ax_context = fig.add_subplot(132, projection='3d')
    set_limits(ax_context, env, args, np_id='value')
    set_labels(ax_context, np_id='value')
    ax_context.set_title('Context and  prediction', pad=25, fontsize=16)

    ax_diff = fig.add_subplot(133, projection='3d')
    set_limits(ax_diff, env, args, np_id='value')
    set_labels(ax_diff, np_id='value')
    ax_diff.set_title('Q - V advantage estimate', pad=25, fontsize=16)
    ax_context.scatter(all_states[0, :, 0].cpu(), all_states[0, :, 1].cpu(), all_rews[0, :, 0].cpu(), c='r',alpha=0.1)
    ax_context.scatter(target_states[0, :, 0].cpu(), target_states[0, :, 1].cpu(), values[0, :, 0].cpu(), c='b',alpha=0.1)
    ax_diff.scatter(target_states[0, :, 0].cpu(), target_states[0, :, 1].cpu(), advantages[0, :, 0].cpu(), c='g', alpha=0.08)
    leg = ax_context.legend(loc="upper right")
    fig.savefig(args.directory_path+'/value/'+name)
    plt.close(fig)


def plot_initial_context(context_points, colors, env, args, i_iter):
    name = 'Contexts of all episodes at iter ' + str(i_iter)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.set_title(name)
    #set_limits(ax, env, args)
    set_labels(ax)
    for e, episode in enumerate(context_points):
        real_len = episode[2]
        z = episode[1][:, :real_len, 0].detach().cpu().numpy()
        xs_context = episode[0][:, :real_len, 0].detach().cpu().numpy()
        ys_context = episode[0][:, :real_len, 1].detach().cpu().numpy()
        ax.scatter(xs_context, ys_context, z, c=colors[e], alpha=0.5)
    fig.savefig(args.directory_path + '/policy/'+ '/All policies samples/' + name)
    plt.close(fig)

def plot_chosen_context(context_list, num_context, i_iter, args, env):
    #colors = []
    num_tested_points = 3
    #for i in range(num_tested_points):
    #    colors.append('#%06X' % randint(0, 0xFFFFFF))
    if args.pick_context:
        if args.pick_dist is None:
            color = 'y'
            name = 'Context chosen by index'
            p=0

        else:
            p=1
            color = 'g'
            name = 'Context chosen by distance'
    else:
        p=2
        color = 'b'
        name = 'All context set'
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(name, fontsize=16)
    test_episode = context_list[0]
    real_len = test_episode[-1]
    e = 0
    for index in np.arange(0, real_len, real_len//(num_tested_points-1)):
        ax = fig.add_subplot(1, 3, e + 1, projection='3d')
        if p==0:
            title = 'Index: {}   number of context points: {}'.format(index, args.num_context)
        elif p==1:
            title = 'Index: {},  euclidian distance: {} '.format(index, args.pick_dist)
        else:
            title = 'Index: {}   number of context points: {}'.format(index, args.num_ensembles*999)
        ax.set_title(title)
        set_limits(ax, env, args, np_id='policy')
        if args.pick_context:
            x_context, y_context = get_close_context(index, test_episode[0][:, index, :].unsqueeze(0), context_list,
                                                     args.pick_dist, num_tot_context=num_context)
        else:
            x_context, y_context = merge_context(context_list)
        if True:
            ax.scatter(x_context[0, :, 0].cpu(), x_context[0, :, 1].cpu(), y_context[0, :, 0].cpu(), c=color, alpha=0.05,
                       marker='+', label='Chosen context', zorder=-1)
            ax.scatter(test_episode[0][0, index, 0].cpu(), test_episode[0][0, index, 1].cpu(),
                       test_episode[1][0, index, 0].cpu(),c='k', alpha=1, label='Target point', zorder=1)
        else:
            ax.scatter(x_context[0, :, 0].cpu(), x_context[0, :, 1].cpu(), y_context[0, :, 0].cpu(), c=color, alpha=0.5,
                       marker='+', zorder=-1)
            ax.scatter(test_episode[0][0, index, 0].cpu(), test_episode[0][0, index, 1].cpu(),
                       test_episode[1][0, index, 0].cpu(),c='k', alpha=1, zorder=1)
        e += 1
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        leg = ax.legend(loc='lower left', bbox_to_anchor=(0.3, 0.2))
        fig.savefig(args.directory_path + '/policy/' + '/All policies samples/' + name + str(i_iter), dpi=300)
    plt.close(fig)