import argparse
import gym
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plotting_functions_RL import *
from core.common import discounted_rewards
from core.agent_picker import AgentPicker
from previous_methods.previous_agents.agent_ensembles_all_context import Agent_all_ctxt
from neural_process import NeuralProcess
from training_module_RL import NeuralProcessTrainerLoo, NeuralProcessTrainerLooPick
from multihead_attention_np import *
from torch.distributions import Normal
from utils.weights_init import InitFunc
# Axes3D import has side effects, it enables using projection='3d' in add_subplot


torch.set_default_tensor_type(torch.DoubleTensor)
if torch.cuda.is_available() and False:
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    device = torch.device("cpu")
print('device: ', device)

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="MountainCarContinuous-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')

parser.add_argument('--use-running-state', default=False,
                    help='store running mean and variance instead of states and actions')
parser.add_argument('--max-kl', type=float, default=0.1, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--num-ensembles', type=int, default=4, metavar='N',
                    help='episode to collect per iteration')
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')

parser.add_argument('--pick-context', default=True, metavar='N',
                    help='pick context points depending on index')
parser.add_argument('--num-context', default=1000, type=int,
                    help='pick context points depending on index')
parser.add_argument('--pick-dist', default=None, type=float,
                    help='if None use index, else defines limit distance for chosing a point')
parser.add_argument('--num-req-steps', type=int, default=1000, metavar='N',
                    help='number of context points to sample from rm')

parser.add_argument('--fixed-sigma', default=0.75, metavar='N', type=float,
                    help='sigma of the policy')
parser.add_argument('--epochs-per-iter', type=int, default=40, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--replay-memory-size', type=int, default=10, metavar='G',
                    help='size of training set in episodes ')
parser.add_argument('--z-dim', type=int, default=100, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--r-dim', type=int, default=128, metavar='N',
                    help='dimension of representation space in np')
parser.add_argument('--h-dim', type=int, default=128, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--np-batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--early-stopping', type=int, default=-100, metavar='N',
                    help='stop training training when avg_loss reaches it')

parser.add_argument('--v-epochs-per-iter', type=int, default=40, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--v-replay-memory-size', type=int, default=2, metavar='G',
                    help='size of training set in episodes')
parser.add_argument('--v-z-dim', type=int, default=128, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--v-r-dim', type=int, default=128, metavar='N',
                    help='dimension of representation space in np')
parser.add_argument('--v-h-dim', type=int, default=128, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--v-np-batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--v-early-stopping', type=int, default=-100, metavar='N',
                    help='stop training training when avg_loss reaches it')

parser.add_argument('--directory-path', default='/home/francesco/PycharmProjects/MasterThesis/NP learning results/',
                    help='path to plots folder')
parser.add_argument('--device-np', default=device,
                    help='device')
parser.add_argument('--dtype', default=torch.float64,
                    help='default type')
parser.add_argument('--seed', type=int, default=7, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')

parser.add_argument('--use-attentive-np', default=True, metavar='N',
                     help='use attention in policy and value NPs')
parser.add_argument('--v-use-attentive-np', default=True, metavar='N',
                     help='use attention in policy and value NPs')
parser.add_argument('--episode-specific-value', default=False, metavar='N',
                    help='condition the value np on all episodes')
parser.add_argument("--plot-every", type=int, default=1,
                    help='plot every n iter')
parser.add_argument("--num-testing-points", type=int, default=1,
                    help='how many point to use as only testing during NP training')
args = parser.parse_args()
initial_training = True
init_func = InitFunc.init_zero

max_episode_len = 999
num_context_points = max_episode_len - args.num_testing_points

np_spec = '_{}z_{}rm_{}vrm_{}e_{}dist_num_context:{}_earlystop{}|{}'.format(args.z_dim, args.replay_memory_size, args.v_replay_memory_size,
                                                       args.epochs_per_iter, args.pick_dist, args.num_context, args.early_stopping, args.v_early_stopping)
run_id = '/V_ABLATION_P_Pick:{}_NP_A_p:{}_A_v:{}_fixSTD:{}_epV:{}_{}ep_{}kl_{}gamma_'.format(args.pick_context, args.use_attentive_np,  args.v_use_attentive_np, args.fixed_sigma, args.episode_specific_value,
                                                args.num_ensembles, args.max_kl, args.gamma) + np_spec
args.directory_path += run_id


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

'''create neural process'''
if args.use_attentive_np:
    policy_np = AttentiveNeuralProcess(state_dim, action_dim, args.r_dim, args.z_dim, args.h_dim,
                                                       args.z_dim, use_self_att=False).to(args.device_np)
else:
    policy_np = NeuralProcess(state_dim, action_dim, args.r_dim, args.z_dim, args.h_dim).to(args.device_np)

optimizer = torch.optim.Adam(policy_np.parameters(), lr=3e-4)
if args.pick_context:
    np_trainer = NeuralProcessTrainerLooPick(args.device_np, policy_np, optimizer, args.pick_dist, args.num_context, print_freq=50)
else:
    np_trainer = NeuralProcessTrainerLoo(args.device_np, policy_np, optimizer,
                                         num_context_range=(num_context_points, num_context_points),
                                         num_extra_target_range=(args.num_testing_points, args.num_testing_points),
                                         print_freq=50)

if args.fixed_sigma is not None:
    args.fixed_sigma = args.fixed_sigma * torch.ones(action_dim)
"""create replay memory"""
replay_memory = ReplayMemoryDataset(args.replay_memory_size, use_mean=True)

"""create agent"""
if args.pick_context:
    agent = AgentPicker(env, policy_np, args.device_np, args.num_context, running_state=running_state, render=args.render,
                  pick_dist=args.pick_dist, fixed_sigma=args.fixed_sigma)
else:
    agent = Agent_all_ctxt(env, policy_np, args.device_np, running_state=running_state, render=args.render,
                  attention=args.use_attentive_np, fixed_sigma=args.fixed_sigma)


def estimate_eta_3(actions, means, advantages, sigmas):
    """Compute learning step from all the samples of previous iteration"""
    d = actions.shape[-1]
    if d > 1:
        raise NotImplementedError('compute eta not implemented for action space of dim>1')
    else:
        stddev = args.fixed_sigma
        iter_sum = 0
        eps = tensor(args.max_kl).to(args.dtype)
        T = tensor(actions.shape[0]).to(args.dtype)
        for action, mean, disc_reward, sigma in zip(actions, means, advantages, sigmas):
            if stddev is None:
                stddev = sigma
            iter_sum += ((disc_reward ** 2) * (action - mean) ** 2) / (2 * (stddev ** 4))
        denominator = iter_sum.to(args.dtype)
        return torch.sqrt((T * eps) / denominator)


def improvement_step_all(complete_dataset, estimated_adv):
    """Perform improvement step using same eta for all episodes"""
    all_improved_context = []
    with torch.no_grad():
        all_states, all_means, all_stdv, all_actions = merge_padded_lists([episode['states'] for episode in complete_dataset],
                                                                [episode['means'] for episode in complete_dataset],
                                                                [episode['stddevs'] for episode in complete_dataset],
                                                                [episode['actions'] for episode in complete_dataset],
                                                                 max_lens=[episode['real_len'] for episode in complete_dataset])
        all_advantages = [adv for ep in estimated_adv for adv in ep]
        eta = estimate_eta_3(all_actions, all_means, all_advantages, all_stdv)
        for episode, episode_adv in zip(complete_dataset, estimated_adv):
            real_len = episode['real_len']
            states = episode['states'][:real_len]
            actions = episode['actions'][:real_len]
            means = episode['means'][:real_len]
            new_padded_actions = torch.zeros_like(episode['actions'])
            new_padded_means = torch.zeros_like(episode['means'])
            i = 0
            for state, action, mean, advantage, stddev in zip(states, actions, means, episode_adv, all_stdv):
                if args.fixed_sigma is None:
                    sigma = stddev
                else:
                    sigma = args.fixed_sigma
                new_mean = mean + eta * advantage * ((action - mean) / sigma)
                distr = Normal(new_mean, sigma)
                new_action = distr.sample()
                new_padded_actions[i, :] = new_action
                new_padded_means[i, :] = new_mean
                i += 1
            episode['new_means'] = new_padded_means
            episode['new_actions'] = new_padded_actions
            if True:
                all_improved_context.append([episode['states'].unsqueeze(0), new_padded_means.unsqueeze(0), real_len])
            else:
                all_improved_context.append([episode['states'].unsqueeze(0), new_padded_actions.unsqueeze(0), real_len])

    return all_improved_context


def train_np(datasets, epochs=args.epochs_per_iter):
    print('Policy training')
    policy_np.training = True
    data_loader = DataLoader(datasets, batch_size=args.np_batch_size, shuffle=True)
    np_trainer.train(data_loader, epochs, early_stopping=args.early_stopping)
    policy_np.training = False


def sample_initial_context_normal(num_episodes):
    initial_episodes = []
    #policy_np.apply(init_func)
    sigma = 0.5
    if args.fixed_sigma is not None:
        sigma = args.fixed_sigma

    for e in range(num_episodes):
        states = torch.zeros([1, max_episode_len, state_dim])

        for i in range(max_episode_len):
            states[:, i, :] = torch.from_numpy(env.observation_space.sample())

        dims = [1, max_episode_len, action_dim]
        distr_init = Normal(zeros(dims), sigma*ones(dims))
        actions_init = distr_init.sample()

        initial_episodes.append([states, actions_init, max_episode_len])
    return initial_episodes

def train_on_initial(initial_context_list):
    #print('training on initial context')
    train_list = []
    for episode in initial_context_list:
        train_list.append([episode[0].squeeze(0), episode[1].squeeze(0), episode[2]])

    train_np(train_list, 2*args.epochs_per_iter)


def create_directories(directory_path):

    os.mkdir(directory_path)
    os.mkdir(directory_path + '/policy/')
    os.mkdir(directory_path + '/value/')
    os.mkdir(directory_path + '/policy/'+'/NP estimate/')
    os.mkdir(directory_path + '/policy/' + '/Mean improvement/')
    os.mkdir(directory_path + '/policy/' + '/Training/')
    os.mkdir(directory_path + '/policy/' + '/All policies samples/')

avg_rewards = [0]
tot_steps = [0]

def main_loop():
    colors = []
    num_episodes = args.num_ensembles
    for i in range(num_episodes):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    improved_context_list = sample_initial_context_normal(args.num_ensembles)
    plot_initial_context(improved_context_list, colors, env, args, '00')
    if initial_training:
        train_on_initial(improved_context_list)
    for i_iter in range(args.max_iter_num):
        print('sampling episodes')
        # (1)
        # generate multiple trajectories that reach the minimum batch_size
        policy_np.training = False
        batch, log = agent.collect_episodes(improved_context_list, args.num_req_steps, args.num_ensembles)  # batch of batch_size transitions from multiple
        #print(log['num_steps'], log['num_episodes'])                # episodes (separated by mask=0). Stored in Memory

        estimated_disc_rew = discounted_rewards(batch.memory, args.gamma)
        complete_dataset = BaseDataset(batch.memory, estimated_disc_rew, args.device_np, args.dtype,  max_len=max_episode_len)

        t0 = time.time()
        improved_context_list = improvement_step_all(complete_dataset, estimated_disc_rew)
        t1 = time.time()
        #plot_initial_context(improved_context_list, colors, env, args, i_iter)
        # plot improved context and actions' discounted rewards
        if i_iter % args.plot_every == 0:
            plot_improvements(complete_dataset, estimated_disc_rew, env, i_iter, args, colors)

        # create training set
        tn0 = time.time()
        replay_memory.add(complete_dataset)
        train_np(replay_memory)
        tn1 = time.time()

        #plot_training_set(i_iter, replay_memory, env, args)
        if i_iter % args.plot_every == 0:
            plot_NP_policy(policy_np, improved_context_list, replay_memory, i_iter, log['avg_reward'], env, args, colors)

        avg_rewards.append(log['avg_reward'])
        tot_steps.append(tot_steps[-1] + log['num_steps'])
        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f} \tT_update {:.4f} \tR_min {:.2f} \tR_max {:.2f} \tR_avg {:.2f}'.format(
                  i_iter, log['sample_time'], t1 - t0, log['min_reward'], log['max_reward'], log['avg_reward']))
            print('Training:  \tT_policy {:.2f}'.format(tn1-tn0))
        if log['avg_reward'] > 95:
            print('converged')
            plot_rewards_history(avg_rewards, args)
        if i_iter % args.plot_every == 0:
            plot_rewards_history(avg_rewards, tot_steps, args)
            #if args.pick_context:
            plot_chosen_context(improved_context_list, args.num_context, i_iter, args, env)
            plot_all_training_set(i_iter, replay_memory, env, args)
        if args.fixed_sigma is not None:
            args.fixed_sigma = args.fixed_sigma * args.gamma
    plot_rewards_history(avg_rewards, args)

    """clean up gpu memory"""
    torch.cuda.empty_cache()

try:
    create_directories(args.directory_path)
except FileExistsError:
    pass

def plot_all_training_set(i_iter, replay_memory, env, args):
    name = 'sTraining trajectories ' + str(i_iter)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    #ax.set_title(name)
    set_limits_v(ax, env, args)
    set_labels(ax)
    xs_context, ys_context = merge_context(replay_memory.data)
    z = ys_context[..., 0].view(-1)
    ax.scatter(xs_context[..., 0], xs_context[..., 1], z, c=z, cmap='viridis', alpha=0.1, vmin=-1., vmax=1.)
    fig.savefig(args.directory_path + '/policy/'+'/Training/'+name)
    plt.close(fig)

main_loop()




