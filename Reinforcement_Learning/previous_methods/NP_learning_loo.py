import argparse
import gym
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plotting_functions_RL import *
from core.common import discounted_rewards, improvement_step_all
from core.agent_picker import AgentPicker
from previous_methods.previous_agents.agent_ensembles_all_context import Agent_all_ctxt
from neural_process import NeuralProcess
from training_module_RL import NeuralProcessTrainerRL, NeuralProcessTrainerLoo, NeuralProcessTrainerLooPick
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
parser.add_argument('--max-kl', type=float, default=0.2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--num-ensembles', type=int, default=10, metavar='N',
                    help='episode to collect per iteration')
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')

parser.add_argument('--pick-context', default=True, metavar='N',
                    help='pick context points depending on index')
parser.add_argument('--num-context', default=10, type=int,
                    help='pick context points depending on index')
parser.add_argument('--pick-dist', default=None, type=float,
                    help='if None use index, else defines limit distance for chosing a point')
parser.add_argument('--rm-as-context', default=True, help='choose subset of rm')
parser.add_argument('--learn-sigma', default=True, help='update the stddev of the policy')


parser.add_argument('--fixed-sigma', default=0.3, metavar='N', type=float,
                    help='sigma of the policy')
parser.add_argument('--epochs-per-iter', type=int, default=30, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--replay-memory-size', type=int, default=50, metavar='G',
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

parser.add_argument('--v-epochs-per-iter', type=int, default=30, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--v-replay-memory-size', type=int, default=50, metavar='G',
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
parser.add_argument("--plot-every", type=int, default=3,
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
run_id = '/VP_freeRM_Pick:{}_NP_A:{}_A_v:{}_fixSTD:{}_epV:{}_{}ep_{}kl_{}gamma_'.format(args.pick_context, args.use_attentive_np,  args.v_use_attentive_np, args.fixed_sigma, args.episode_specific_value,
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
args.fixed_sigma = args.fixed_sigma * torch.ones(action_dim)

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
                                         num_target=args.num_testing_points,
                                         print_freq=50)

if args.v_use_attentive_np:
    value_np = AttentiveNeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_h_dim,
                                      args.z_dim, use_self_att=False).to(args.device_np)
else:
    value_np = NeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_h_dim).to(args.device_np)
value_optimizer = torch.optim.Adam(value_np.parameters(), lr=3e-4)
value_np_trainer = NeuralProcessTrainerLoo(args.device_np, value_np, value_optimizer,
                                          num_target=args.num_testing_points,
                                          print_freq=50)
value_np.training = False
"""create replay memory"""
replay_memory = ReplayMemoryDataset(args.replay_memory_size)
value_replay_memory = ValueReplay(args.v_replay_memory_size)

"""create agent"""
if args.pick_context:
    agent = AgentPicker(env, policy_np, args.device_np, args.num_context, running_state=running_state, render=args.render,
                  pick_dist=args.pick_dist, fixed_sigma=args.fixed_sigma)
else:
    agent = Agent_all_ctxt(env, policy_np, args.device_np, running_state=running_state, render=args.render,
                  attention=args.use_attentive_np, fixed_sigma=args.fixed_sigma)


def train_np(datasets, epochs=args.epochs_per_iter):
    print('Policy training')
    policy_np.training = True
    data_loader = DataLoader(datasets, batch_size=args.np_batch_size, shuffle=True)
    np_trainer.train(data_loader, epochs, early_stopping=args.early_stopping)
    policy_np.training = False


def train_value_np(value_replay_memory):
    print('Value training')
    value_np.training = True
    value_data_loader = DataLoader(value_replay_memory, batch_size=args.v_np_batch_size, shuffle=True)
    value_np_trainer.train(value_data_loader, args.v_epochs_per_iter, early_stopping=args.v_early_stopping)
    value_np.training = False


def estimate_v_a(complete_dataset, disc_rew, env, args, i_iter):
    ep_rewards = disc_rew
    ep_states = [ep['states'] for ep in complete_dataset]
    real_lens = [ep['real_len'] for ep in complete_dataset]
    estimated_advantages = []
    all_values = []
    if not len(value_replay_memory.data) == 0:
        s_context, r_context = merge_context(value_replay_memory.data)
        for states, rewards, real_len in zip(ep_states, ep_rewards, real_lens):
            s_target = states[:real_len, :].unsqueeze(0)
            r_target = rewards.view(1, -1, 1)
            with torch.no_grad():
                values = value_np(s_context, r_context, s_target)
            advantages = r_target - values.mean
            estimated_advantages.append(advantages.squeeze(0))
            all_values.append(values.mean)
    else:
        for i in range(len(ep_states)):
            context_list = []
            j = 0
            for states, rewards, real_len in zip(ep_states, ep_rewards, real_lens):
                if j != i:
                    context_list.append([states.unsqueeze(0), rewards.view(1,-1,1), real_len])
                else:
                    s_target = states[:real_len, :].unsqueeze(0)
                    r_target = rewards.view(1, -1, 1)
                j += 1
            s_context, r_context = merge_context(context_list)
            with torch.no_grad():
                values = value_np(s_context, r_context, s_target)
            advantages = r_target - values.mean
            estimated_advantages.append(advantages.squeeze(0))
            all_values.append(values.mean)

    plot_NP_value(value_np, s_context, s_target, values.mean, r_context, advantages, env, args, i_iter)
    return estimated_advantages


def sample_initial_context_normal(num_episodes):
    initial_episodes = []
    #policy_np.apply(init_func)
    sigma = args.fixed_sigma*0.1

    for e in range(num_episodes):
        states = torch.zeros([1, max_episode_len, state_dim])

        for i in range(max_episode_len):
            states[:, i, :] = torch.randn(state_dim) #torch.from_numpy(env.observation_space.sample())

        if args.use_attentive_np or True:
            dims = [1, max_episode_len, action_dim]
            distr_init = Normal(zeros(dims), sigma*ones(dims))
            actions_init = distr_init.sample()
        else:
            z_sample = torch.randn((1, args.z_dim)).unsqueeze(1).repeat(1, max_episode_len, 1)
            means_init, stds_init = policy_np.xz_to_y(states, z_sample)
            actions_init = Normal(means_init, stds_init).sample()
        initial_episodes.append([states, actions_init, max_episode_len])
    return initial_episodes


def train_on_initial(initial_context_list):
    #print('training on initial context')
    train_list = []
    for episode in initial_context_list:
        train_list.append([episode[0].squeeze(0), episode[1].squeeze(0), episode[2]])

    policy_np.training = True
    data_loader = DataLoader(train_list, batch_size=args.np_batch_size, shuffle=True)
    np_trainer.train(data_loader, 10*args.epochs_per_iter, early_stopping=0)
    policy_np.training = False

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
    if initial_training:
        train_on_initial(improved_context_list)
    for i_iter in range(args.max_iter_num):
        print('sampling episodes')
        # generate multiple trajectories that reach the minimum batch_size
        if len(replay_memory) == 0 or not args.rm_as_context:
            context_list_np = improved_context_list
        else:
            context_list_np = replay_memory.data
        policy_np.training = False
        batch, log = agent.collect_episodes(context_list_np, args.num_ensembles)  # batch of batch_size transitions from multiple
        #print(log['num_steps'], log['num_episodes'])                # episodes (separated by mask=0). Stored in Memory

        disc_rew = discounted_rewards(batch.memory, args.gamma)
        complete_dataset = BaseDataset(batch.memory, disc_rew, args.device_np, args.dtype,  max_len=max_episode_len)

        advantages_np = estimate_v_a(complete_dataset, disc_rew, env, args, i_iter)
        improved_context_list_np = improvement_step_all(complete_dataset, advantages_np, args.max_kl, args)

        tv0 = time.time()
        value_replay_memory.add(complete_dataset)
        train_value_np(value_replay_memory)
        tv1 = time.time()
        # plot improved context and actions' discounted rewards
        if i_iter % args.plot_every == 0:
            plot_improvements(complete_dataset, advantages_np, env, i_iter, args, colors)
            #plot_initial_context(improved_context_list, colors, env, args, i_iter)

        # create training set
        tn0 = time.time()
        replay_memory.add(complete_dataset)
        train_np(replay_memory)
        tn1 = time.time()

        #plot_training_set(i_iter, replay_memory, env, args)
        if i_iter % args.plot_every == 0:
            plot_NP_policy(policy_np, improved_context_list, replay_memory, i_iter, log['avg_reward'], env, args, colors)

        avg_rewards.append(log['avg_reward'])
        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}  \tR_min {:.2f} \tR_max {:.2f} \tR_avg {:.2f}'.format(
                  i_iter, log['sample_time'], log['min_reward'], log['max_reward'], log['avg_reward']))
            print('Training:  \tT_policy {:.2f}  \tT_value {:.2f}'.format(tn1-tn0, tv1-tv0))
        tot_steps.append(tot_steps[-1] + log['num_steps'])
        avg_rewards.append(log['avg_reward'])
        tot_steps.append(log['num_steps'])
        if i_iter % args.log_interval == 0:
            print(i_iter)
            print('np: \tR_min {:.2f} \tR_max {:.2f} \tR_avg {:.2f}'.format(log['min_reward'], log['max_reward'],
                                                                            log['avg_reward']))
        print('new sigma', args.fixed_sigma)
        plot_rewards_history(tot_steps, avg_rewards)
        #store_avg_rewards(tot_steps[-1], avg_rewards[-1],
        #                  np_file.replace(str(args.seed) + '.csv', 'avg' + str(args.seed) + '.csv'))
        if tot_steps[-1] > args.tot_steps:
            break
    """clean up gpu memory"""
    torch.cuda.empty_cache()


def plot_rewards_history(steps, rews):
    fig_rew, ax_rew = plt.subplots(1, 1)

    ax_rew.plot(steps[1:], rews[1:], c='b', label='AttentiveNP')
    ax_rew.set_xlabel('number of steps')
    ax_rew.set_ylabel('average reward')
    ax_rew.set_title('Average Reward History')
    plt.legend()
    plt.grid()
    fig_rew.savefig(args.directory_path + run_id.replace('.', ',')+str(args.seed))
    plt.close(fig_rew)
try:
    create_directories(args.directory_path)
except FileExistsError:
    pass
main_loop()




