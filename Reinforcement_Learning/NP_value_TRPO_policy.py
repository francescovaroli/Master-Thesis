import argparse
import gym
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multihead_attention_np import *
from utils_rl import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from core.common import discounted_rewards
from core.trpo import trpo_step
from core.common import estimate_advantages
from core.agent_NN import Agent
from neural_process import NeuralProcess
from training_module_RL import NeuralProcessTrainerRL
from training_leave_one_out import NeuralProcessTrainerLoo
from new_plotting_functions import *
torch.set_default_tensor_type(torch.DoubleTensor)

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="MountainCarContinuous-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--num-threads', type=int, default=2, metavar='N',
                    help='number of threads for training (default: 4)')
parser.add_argument('--seed', type=int, default=7, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=4994, metavar='N',
                    help='minimal batch size per TRPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=501, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--use-running-state', default=False,
                    help='store running mean and variance instead of states and actions')

parser.add_argument('--v-epochs-per-iter', type=int, default=20, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--v-replay-memory-size', type=int, default=60, metavar='G',
                    help='size of training set in episodes')
parser.add_argument('--v-z-dim', type=int, default=128, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--v-r-dim', type=int, default=128, metavar='N',
                    help='dimension of represenation space in np')
parser.add_argument('--v-h-dim', type=int, default=128, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--v-np-batch-size', type=int, default=8, metavar='N',
                    help='batch size for np training')
parser.add_argument('--use-attentive-np', default=False, metavar='N',
                     help='use attention in policy and value NPs')
parser.add_argument('--episode-specific-value', default=False, metavar='N',
                    help='condition the value np on all episodes')
parser.add_argument("--plot-every", type=int, default=1,
                    help='plot every n iter')
parser.add_argument("--num-testing-points", type=int, default=1,
                    help='how many point to use as only testing during NP training')
parser.add_argument('--device', default=torch.device('cpu'),
                    help='device')
parser.add_argument('--directory-path', default='/home/francesco/PycharmProjects/MasterThesis/value learning/',
                    help='path to plots folder')
parser.add_argument('--device-np', default=torch.device('cpu'),
                    help='device')
parser.add_argument('--dtype', default=torch.float64,
                    help='default type')
parser.add_argument('--loo', default=False, metavar='N',
                     help='train with leave-one-out method')
args = parser.parse_args()

dtype = torch.float64
device = args.device

process_epochs = args.v_epochs_per_iter//args.num_processes
max_episode_len = 999
num_context_points = max_episode_len - args.num_testing_points

np_spec = '_{}z_{}rm_{}e_num_context:{}/'.format(args.v_z_dim, args.v_replay_memory_size,
                                                       args.v_epochs_per_iter, num_context_points)
run_id = '/Value_NP_deep2_A:{}_epV:{}_{}bsize_{}kl_{}gamma_'.format(args.use_attentive_np,  args.episode_specific_value,
                                                                 args.min_batch_size, args.max_kl, args.gamma) + np_spec
args.directory_path += run_id

torch.set_default_dtype(dtype)

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
if args.use_running_state:
    running_state = ZFilter((state_dim,), clip=5)  # running list of states that allows to access precise mean and std
else:
    running_state = None

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

if args.use_attentive_np:
    value_np = AttentiveNeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_h_dim,
                                      args.z_dim, use_self_att=False).to(args.device)
else:
    value_np = NeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_h_dim).to(args.device)
value_optimizer = torch.optim.Adam(value_np.parameters(), lr=3e-4)
if args.loo:
    value_np_trainer = NeuralProcessTrainerLoo(args.device, value_np, value_optimizer,
                                              num_context_range=(num_context_points, num_context_points),
                                              num_extra_target_range=(args.num_testing_points, args.num_testing_points),
                                              print_freq=50)
else:
    value_np_trainer = NeuralProcessTrainerRL(args.device, value_np, value_optimizer,
                                              num_context_range=(num_context_points, num_context_points),
                                              num_extra_target_range=(args.num_testing_points, args.num_testing_points),
                                              print_freq=50)

value_replay_memory = ValueReplay(args.v_replay_memory_size)

policy_net = Policy(state_dim, action_dim, log_std=args.log_std)
policy_net.to(args.device)

value_net = Value(state_dim)

"""create agent"""
agent = Agent(env, policy_net, args.device, running_state=running_state, render=args.render, num_threads=args.num_threads)


def update_params_trpo(batch, i_iter):
    # (3)
    value_np.training = False
    num_transition = 0
    idx = []
    for b in batch:
        l = len(b)
        idx.append(l+num_transition-1)
        num_transition += l
    masks = ones(num_transition)
    masks[idx] = 0
    states = zeros(1, num_transition, state_dim)
    actions = zeros(1, num_transition, action_dim)
    disc_rewards = zeros(1, num_transition, 1)
    rewards = zeros(1, num_transition, 1)
    i = 0
    for e, ep in enumerate(batch):
        for t, tr in enumerate(ep):
            states[:, i, :] = torch.from_numpy(tr.state).to(dtype).to(device)
            actions[:, i, :] = torch.from_numpy(tr.action).to(dtype).to(device)
            disc_rewards[:, i, :] = torch.tensor(tr.disc_rew).to(dtype).to(device)
            rewards[:, i, :] = torch.tensor(tr.reward).to(dtype).to(device)
            i += 1

    with torch.no_grad():
        values_distr = value_np(states, disc_rewards, states)  # estimate value function of each state with NN
        values = values_distr.mean

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards.squeeze_(0), masks, values.squeeze_(0), args.gamma, args.tau, device)

    """plot"""
    plot_values(value_np, value_net, states, values, advantages, disc_rewards, env, args, i_iter)

    """perform TRPO update"""
    trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg)

def train_value_np(value_replay_memory):
    print('Value training')
    value_np.training = True
    value_data_loader = DataLoader(value_replay_memory, batch_size=args.v_np_batch_size, shuffle=True)
    value_np_trainer.train(value_data_loader, process_epochs, early_stopping=0)
    value_np.training = False


def create_directories(directory_path):
    os.mkdir(directory_path)


def main_loop():
    for i_iter in range(args.max_iter_num):
        # (1)
        # generate multiple trajectories that reach the minimum batch_size
        # introduce param context=None when np is policy, these will be the context points used to predict
        batch, log, memory = agent.collect_samples(args.min_batch_size)  # batch of batch_size transitions from multiple
        print(log['num_steps'], log['num_episodes'])                     # episodes (separated by mask=0). Stored in Memory
        disc_rew = discounted_rewards(batch, args.gamma)
        memory.set_disc_rew(disc_rew)
        complete_dataset = BaseDataset(batch, disc_rew, args.device, dtype,  max_len=max_episode_len)

        t0 = time.time()
        update_params_trpo(batch, i_iter)  # estimate advantages from samples and update policy by TRPO step
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1 - t0, log['min_reward'], log['max_reward'], log['avg_reward']))

        value_replay_memory.add(complete_dataset)
        train_value_np(value_replay_memory)


def plot_values(value_np, value_net, all_states, all_values, all_advantages, all_rews, env, args, i_iter):
    name = 'Value estimate' + str(i_iter)
    fig = plt.figure(figsize=(16, 14))  # figsize=plt.figaspect(1.5)
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
    vmin = stddev_low.min()
    vmax = stddev_high.max()
    ax_mean = fig.add_subplot(221, projection='3d')
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

    ax_mean.set_title('Mean and std. dev. of the value NP', pad=20, fontsize=16)

    '''ax_nn = fig.add_subplot(222, projection='3d')
    set_limits(ax_nn, env, args, np_id='value')
    set_labels(ax_nn, np_id='value')
    ax_nn.set_title('Mean and std. dev. of the value NN', pad=20, fontsize=14)
    zs = []
    for _x1 in x1:
        for _x2 in x2:
            state = tensor([_x1, _x2], device=args.device).unsqueeze(0)
            zs.append(value_net(state)[0][0].item())
    Z = (np.array(zs)).reshape(X1.shape).transpose()
    ax_nn.plot_surface(X1, X2, Z, cmap='viridis', vmin=-1., vmax=1.)'''

    ax_context = fig.add_subplot(222, projection='3d')
    set_limits(ax_context, env, args, np_id='value')
    set_labels(ax_context, np_id='value')
    ax_context.set_title('Context and  prediction', pad=20, fontsize=16)

    ax_diff = fig.add_subplot(223, projection='3d')
    set_limits(ax_diff, env, args, np_id='value')
    set_labels(ax_diff, np_id='value')
    ax_diff.set_title('Q - V advantage estimate')
    diff = all_rews.squeeze(0) - all_values
    ax_diff.scatter(all_states[0, :, 0].detach(), all_states[0, :, 1].detach(), diff[:,0].detach())

    ax_adv = fig.add_subplot(224, projection='3d')
    set_labels(ax_adv, np_id='value')
    set_limits(ax_adv, env, args, np_id='value')
    ax_adv.set_title('GAE advantage estimate')

    ax_context.scatter(all_states[0, :, 0].detach(), all_states[0, :, 1].detach(), all_rews[0, :, 0].detach(), c='r',
               label='Discounted rewards (context)', alpha=0.1)
    ax_context.scatter(all_states[0, :, 0].detach(), all_states[0, :, 1].detach(), all_values[:, 0].detach(), c='b',
               label='Estimated values', alpha=0.1)

    leg = ax_context.legend(loc="upper right")
    ax_adv.scatter(all_states[0, :, 0].detach(), all_states[0, :, 1].detach(), all_advantages[:, 0].detach(), c='g', label='R-V')
    fig.savefig(args.directory_path+name)
    plt.close(fig)

create_directories(args.directory_path)
main_loop()
