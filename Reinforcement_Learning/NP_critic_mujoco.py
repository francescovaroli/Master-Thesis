import argparse
import gym
import os
import sys
import time
from random import randint
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_rl.torch import *
from utils_rl.memory_dataset import *
from utils_rl.store_results import *
from core.agent_samples_all_context import Agent_all_ctxt
#from core.agent_conditionning import Agent_all_ctxt
from core.agent_picker import AgentPicker
from neural_process import NeuralProcess
from training_leave_one_out import NeuralProcessTrainerLoo
from training_module_RL import NeuralProcessTrainerRL
from training_leave_one_out_pick import NeuralProcessTrainerLooPick
import scipy.optimize
from models.mlp_critic import Value
from env_wrappers import AntWrapper, HumanoidWrapper, InvertedDoublePendulumWrapper
from multihead_attention_np import *
from torch.distributions import Normal
from core.common import discounted_rewards, estimate_v_a, improvement_step_all, compute_gae


torch.set_default_tensor_type(torch.DoubleTensor)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    device = torch.device("cpu")
print('device: ', device)

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="HumanoidStandup-v2", metavar='G',
                    help='name of the environment')
parser.add_argument('--render', default=False, type=boolean_string,
                    help='render the environment')
parser.add_argument('--mean-action', default=False, type=boolean_string, help='update the stddev of the policy')

parser.add_argument('--learn-sigma', default=True, type=boolean_string, help='update the stddev of the policy')
parser.add_argument('--loo', default=True, type=boolean_string, help='train leaving episode out')
parser.add_argument('--pick', default=True, type=boolean_string, help='choose subset of rm')
parser.add_argument('--rm-as-context', default=True, type=boolean_string, help='choose subset of rm')
parser.add_argument('--gae', default=True, type=boolean_string, help='use generalized advantage estimate')
parser.add_argument('--learn-baseline', default=True, type=boolean_string, help='use V estimate')

parser.add_argument('--value-net', default=True, type=boolean_string, help='use NN for V estimate')


parser.add_argument('--num-context', type=int, default=5000, metavar='N',
                    help='number of context points to sample from rm')
parser.add_argument('--num-req-steps', type=int, default=1000, metavar='N',
                    help='number steps')

parser.add_argument('--use-running-state', default=False, type=boolean_string,
                    help='store running mean and variance instead of states and actions')
parser.add_argument('--max-kl-np', type=float, default=.2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--num-ensembles', type=int, default=5, metavar='N',
                    help='episode to collect per iteration')
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='discount factor (default: 0.95)')

parser.add_argument('--fixed-sigma', default=0.3, type=float, metavar='N',
                    help='sigma of the policy')
parser.add_argument('--min-sigma', default=0.1, type=float, metavar='N',
                    help='minimum value of the NP policy stddev')
parser.add_argument('--epochs-per-iter', type=int, default=20, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--replay-memory-size', type=int, default=20, metavar='G',
                    help='size of training set in episodes ')
parser.add_argument('--z-dim', type=int, default=32, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--r-dim', type=int, default=32, metavar='N',
                    help='dimension of representation space in np')
parser.add_argument('--h-dim', type=int, default=32, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--a-dim', type=int, default=32, metavar='N',
                    help='dimension of representation space in np')
parser.add_argument('--np-batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--early-stopping', type=int, default=-100000, metavar='N',
                    help='stop training training when avg_loss reaches it')

parser.add_argument('--v-epochs-per-iter', type=int, default=20, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--v-replay-memory-size', type=int, default=100, metavar='G',
                    help='size of training set in episodes')
parser.add_argument('--v-z-dim', type=int, default=128, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--v-r-dim', type=int, default=128, metavar='N',
                    help='dimension of representation space in np')

parser.add_argument('--v-np-batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--v-early-stopping', type=int, default=-100000, metavar='N',
                    help='stop training training when avg_loss reaches it')

parser.add_argument('--directory-path', default='/home/francesco/PycharmProjects/MasterThesis/mujoco learning results/',
                    help='path to plots folder')
parser.add_argument('--tot-steps', default=1000000, type=int,
                    help='total steps in the run')


parser.add_argument('--device-np', default=device,
                    help='device')
parser.add_argument('--dtype', default=torch.float64,
                    help='default type')
parser.add_argument('--seed', type=int, default=0, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')

parser.add_argument('--use-attentive-np', default=True, type=boolean_string, metavar='N',
                     help='use attention in policy and value NPs')
parser.add_argument('--v-use-attentive-np', default=True, type=boolean_string, metavar='N',
                     help='use attention in policy and value NPs')
parser.add_argument('--episode-specific-value', default=False, type=boolean_string, metavar='N',
                    help='condition the value np on all episodes')
parser.add_argument("--plot-every", type=int, default=1,
                    help='plot every n iter')
parser.add_argument("--num-testing-points", type=int, default=1000,
                    help='how many point to use as only testing during NP training')

parser.add_argument("--net-size", type=int, default=2,
                    help='multiplies all net pararms')

args = parser.parse_args()
initial_training = True

args.z_dim *= args.net_size
args.r_dim *= args.net_size
args.h_dim *= args.net_size
args.a_dim *= args.net_size

args.epochs_per_iter = min(2000 // args.replay_memory_size, 50)
args.v_epochs_per_iter = args.epochs_per_iter
args.v_replay_memory_size = args.replay_memory_size
args.v_z_dim = args.z_dim
args.v_r_dim = args.r_dim

np_spec = '_NP_baseline:{}_critic:{}_{},{}rm_isctxt:{}_{},{}epo_{}z_{}h_{}kl_attention:{}_{}a'.format(args.learn_baseline, args.value_net, args.replay_memory_size, args.v_replay_memory_size,
                                                                args.rm_as_context, args.epochs_per_iter,args.v_epochs_per_iter, args.z_dim,
                                                                args.h_dim, args.max_kl_np, args.use_attentive_np,
                                                                                args.a_dim)


run_id = '/{}_0_NP_{}steps_{}epi_fixSTD:{}_min_sigma:{}_{}gamma' \
         '_{}target_loo:{}_picklc:{}_{}context'.format(args.env_name, args.num_req_steps, args.num_ensembles,
                                                            args.fixed_sigma, args.min_sigma, args.gamma, args.num_testing_points, args.loo,
                                                            args.pick, args.num_context) + np_spec
run_id = run_id.replace('.', ',')
args.directory_path += run_id

np_file = args.directory_path + '/{}.csv'.format(args.seed)

#torch.set_default_dtype(args.dtype)

"""environment"""
if args.env_name == 'Humanoid-v2' or args.env_name == 'HumanoidStandup-v2':
    env = HumanoidWrapper(args.env_name)
elif args.env_name == 'Ant-v2':
    env = AntWrapper(args.env_name)
elif args.env_name == 'InvertedDoublePendulum-v2':
    env = InvertedDoublePendulumWrapper(args.env_name)
else:
    env = gym.make(args.env_name)
max_episode_len = env._max_episode_steps

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print('state_dim', state_dim)
print('action_dim', action_dim)
is_disc_action = len(env.action_space.shape) == 0

if args.fixed_sigma is not None:
    args.fixed_sigma = args.fixed_sigma * torch.ones(action_dim)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)


'''create neural process'''
if args.use_attentive_np:
    policy_np = AttentiveNeuralProcess(state_dim, action_dim, args.r_dim, args.z_dim, args.h_dim,
                                                       args.a_dim, use_self_att=False, min_sigma=args.min_sigma).to(args.device_np)
else:
    policy_np = NeuralProcess(state_dim, action_dim, args.r_dim, args.z_dim, args.h_dim, min_sigma=args.min_sigma).to(args.device_np)
optimizer = torch.optim.Adam(policy_np.parameters(), lr=3e-4)

if args.value_net:
    value_net = Value(state_dim)
    value_net.to(args.device_np)
else:
    if args.v_use_attentive_np:
        value_np = AttentiveNeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_r_dim,
                                          args.a_dim, use_self_att=False).to(args.device_np)
    else:
        value_np = NeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_h_dim).to(args.device_np)
    value_optimizer = torch.optim.Adam(value_np.parameters(), lr=3e-4)
    if args.pick:
        value_np_trainer = NeuralProcessTrainerLooPick(args.device_np, value_np, value_optimizer, pick_dist=None,
                                                       num_context=args.num_context)
    else:
        value_np_trainer = NeuralProcessTrainerLoo(args.device_np, value_np, value_optimizer,
                                                   num_target=args.num_testing_points,
                                                   print_freq=50)
    value_np.training = False

if args.loo and (not args.pick):
    print('using Loo trainer')
    np_trainer = NeuralProcessTrainerLoo(args.device_np, policy_np, optimizer, num_target=args.num_testing_points,
                                         print_freq=50)
if args.loo and args.pick:
    print('using LooPick trainer')
    np_trainer = NeuralProcessTrainerLooPick(args.device_np, policy_np, optimizer, pick_dist=None,
                                            num_context=args.num_context)
if not args.loo:
    print('using RL trainer')
    np_trainer = NeuralProcessTrainerRL(args.device_np, policy_np, optimizer, (1, max_episode_len//2),
                                        print_freq=50)


"""create replay memory"""
replay_memory = ReplayMemoryDataset(args.replay_memory_size)
value_replay_memory = ValueReplay(args.v_replay_memory_size)

"""create agent"""
if not args.pick:
    print('agent all')
    agent_np = Agent_all_ctxt(env, policy_np, args.device_np, running_state=None, render=args.render,
                              attention=args.use_attentive_np, fixed_sigma=args.fixed_sigma, mean_action=args.mean_action)
else:
    print('agent pick')
    agent_np = AgentPicker(env, policy_np, args.device_np, args.num_context, custom_reward=None, pick_dist=None,
                           mean_action=False, render=args.render, running_state=None, fixed_sigma=args.fixed_sigma)

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


def sample_initial_context_normal(num_episodes):
    initial_episodes = []
    #policy_np.apply(init_func)
    if args.fixed_sigma is not None:
        sigma = args.fixed_sigma*0.1
    else:
        sigma = 0.1*ones(action_dim)
    for e in range(2):
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


def critic_estimate(states_list, rewards_list, args):
    adv_list = []
    ret_list = []
    for states, rewards in zip(states_list, rewards_list):
        with torch.no_grad():
            values = value_net(states)
        advantages = compute_gae(rewards, values, args.gamma, args.tau)
        returns = values + advantages
        adv_list.append(advantages)
        ret_list.append(returns)
    return adv_list, ret_list


def update_critic(states, returns, l2_reg=1e-3):
    def get_value_loss(flat_params):
        """
        compute the loss for the value network comparing estimated values with empirical ones
        and optimizes the network parameters
        :param flat_params:
        :return:
        """
        set_flat_params_to(value_net, tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()  # MeanSquaredError

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return value_loss.item(), get_flat_grad_from(value_net.parameters()).cpu().numpy()

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                            get_flat_params_from(value_net).detach().cpu().numpy(),
                                                            maxiter=25)
    set_flat_params_to(value_net, tensor(flat_params))


avg_rewards_np = [0]
tot_steps_np = [0]


def main_loop():
    colors = []
    num_episodes = args.num_ensembles
    for i in range(num_episodes):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    improved_context_list_np = sample_initial_context_normal(args.num_ensembles)
    for i_iter in range(args.max_iter_num):

        # generate multiple trajectories that reach the minimum batch_size
        policy_np.training = False
        if len(replay_memory) == 0 or not args.rm_as_context:
            context_list_np = improved_context_list_np
        else:
            context_list_np = replay_memory.data
        batch_np, log_np = agent_np.collect_episodes(context_list_np, args.num_req_steps, args.num_ensembles)

        disc_rew_np = discounted_rewards(batch_np.memory, args.gamma)
        iter_dataset_np = BaseDataset(batch_np.memory, disc_rew_np, args.device_np, args.dtype,  max_len=max_episode_len)
        print('np avg actions: ', log_np['action_mean'])
        if args.learn_baseline:
            if args.value_net:
                state_list = [ep['states'][:ep['real_len']] for ep in iter_dataset_np]
                advantages_np, returns = critic_estimate(state_list, disc_rew_np, args)
                update_critic(torch.cat(state_list, dim=0), torch.cat(returns, dim=0))
            else:
                advantages_np = estimate_v_a(iter_dataset_np, disc_rew_np, value_replay_memory, value_np, args)
                value_replay_memory.add(iter_dataset_np)
                train_value_np(value_replay_memory)
        else:
            advantages_np = disc_rew_np
        improved_context_list_np = improvement_step_all(iter_dataset_np, advantages_np, args.max_kl_np, args)
        # training
        tn0 = time.time()
        replay_memory.add(iter_dataset_np)
        train_np(replay_memory)
        tn1 = time.time()
        tot_steps_np.append(tot_steps_np[-1] + log_np['num_steps'])
        avg_rewards_np.append(log_np['avg_reward'])

        if i_iter % args.log_interval == 0:
            print(i_iter)
            print('np: \tR_min {:.2f} \tR_max {:.2f} \tR_avg {:.2f}'.format(log_np['min_reward'], log_np['max_reward'], log_np['avg_reward']))
        print('new sigma', args.fixed_sigma)
        plot_rewards_history(tot_steps_np, avg_rewards_np)
        store_avg_rewards(tot_steps_np[-1], avg_rewards_np[-1], np_file.replace(str(args.seed)+'.csv', 'avg'+str(args.seed)+'.csv'))
        if tot_steps_np[-1] > args.tot_steps or log_np['avg_reward'] < -5000:
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


def create_directories(directory_path):
    os.mkdir(directory_path)

try:
    create_directories(args.directory_path)
except FileExistsError:
    pass

main_loop()
