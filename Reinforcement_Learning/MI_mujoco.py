import argparse
import gym
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_rl.torch import *
from utils_rl.memory_dataset import *
from RL_results.store_results import *

from core.agent_samples_all_context import Agent_all_ctxt
from core.agent_picker import AgentPicker
from MeanInterpolatorModel import MeanInterpolator, MITrainer
from multihead_attention_np import *

from core.common import estimate_v_a, improvement_step_all, discounted_rewards, sample_initial_context_normal, critic_estimate, update_critic
from core.mlp_critic import Value
from utils_rl.env_wrappers import AntWrapper, HumanoidWrapper, InvertedDoublePendulumWrapper


torch.set_default_tensor_type(torch.DoubleTensor)
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    device = torch.device("cpu")
print('device: ', device)

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="Humanoid-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')

parser.add_argument('--learn-sigma', default=True, type=boolean_string,
                    help='update the stddev of the policy (update step)')
parser.add_argument('--fixed-sigma', default=0.35, type=float, metavar='N',
                    help='sigma of the policy')

parser.add_argument('--loo', default=True, type=boolean_string, help='train leaving episode out')
parser.add_argument('--pick', default=False, type=boolean_string, help='choose subset of rm')
parser.add_argument('--num-context', type=int, default=2500, metavar='N',
                    help='number of context points to sample from rm')
parser.add_argument('--rm-as-context', default=True, type=boolean_string,
                    help='choose subset of rm')

parser.add_argument('--value-net', default=True, type=boolean_string, help='use NN for V estimate (actor-critic)')
parser.add_argument('--gae', default=True, type=boolean_string, help='use generalized advantage estimate ')
parser.add_argument('--tau', type=float, default=0.9, metavar='G',
                    help='discount factor (default: 0.95)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')

parser.add_argument('--num-req-steps', type=int, default=5000, metavar='N',
                    help='min number of steps to gather in one iter')
parser.add_argument('--use-running-state', default=False, type=boolean_string,
                    help='store running mean and variance instead of states and actions')
parser.add_argument('--max-kl-mi', type=float, default=1.5, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--num-ensembles', type=int, default=10, metavar='N',
                    help='min num episode to collect per iteration')
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 500)')


# MKI specs
parser.add_argument('--z-mi-dim', type=int, default=27, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--h-mi-dim', type=int, default=64, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--scaling', default='uniform', metavar='N',
                    help='feature extractor scaling')
parser.add_argument("--lr_nn", type=float, default=1e-4,
                    help='plot every n iter')
parser.add_argument('--epochs-per-iter', type=int, default=60, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--replay-memory-size', type=int, default=50, metavar='G',
                    help='size of training set in episodes ')
parser.add_argument('--early-stopping', type=int, default=-1000, metavar='N',
                    help='stop training training when avg_loss reaches it')
parser.add_argument("--num-testing-points", type=int, default=1000,
                    help='how many point to use as only testing during MI training')
parser.add_argument("--net-size", type=int, default=1,
                    help='multiplies all net pararms')

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
parser.add_argument("--plot-every", type=int, default=1,
                    help='plot every n iter')


args = parser.parse_args()
initial_training = True

args.epochs_per_iter = 1000 // args.replay_memory_size

args.z_mi_dim *= args.net_size
args.h_mi_dim *= args.net_size


mi_spec = '_MI_critic:{}_{}_steps_{}rm_isctxt:{}_{}epo_{}z_{}h_{}kl_{}'.format(args.value_net, args.num_req_steps, args.replay_memory_size, args.rm_as_context, args.epochs_per_iter, args.z_mi_dim,
                                          args.h_mi_dim,args.max_kl_mi, args.scaling)

run_id = '/{}_MI_{}epi_fixSTD:{}_{}gamma_{}target_loo:{}_pick:{}_{}context'.format(args.env_name, args.num_ensembles, args.fixed_sigma,
                                                                    args.gamma, args.num_testing_points, args.loo,
                                                                                   args.pick, args.num_context) + mi_spec
run_id = run_id.replace('.', ',')
args.directory_path += run_id

mi_file = args.directory_path + '/{}.csv'.format(args.seed)

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
is_disc_action = len(env.action_space.shape) == 0
print('state_dim', state_dim)
print('action_dim', action_dim)

args.fixed_sigma = args.fixed_sigma * torch.ones(action_dim)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

# MKI model
model = MeanInterpolator(state_dim, args.h_mi_dim, args.z_mi_dim, scaling=args.scaling).to(device).double()

optimizer_mi = torch.optim.Adam([
    {'params': model.feature_extractor.parameters(), 'lr': args.lr_nn},
    {'params': model.interpolator.parameters(), 'lr': args.lr_nn}])


# trainer
model_trainer = MITrainer(device, model, optimizer_mi, num_context=args.num_context, num_target=args.num_testing_points, print_freq=50)

if args.value_net:
    value_net = Value(state_dim)
    value_net.to(args.device_np)

# RM
replay_memory_mi = ReplayMemoryDataset(args.replay_memory_size)
value_replay_memory = ValueReplay(args.replay_memory_size)

"""create agent"""
agent_mi = Agent_all_ctxt(env, model, args.device_np, running_state=None, render=args.render, fixed_sigma=args.fixed_sigma)
if not args.pick:
    agent_mi = Agent_all_ctxt(env, model, args.device_np, running_state=None, render=args.render,
                              fixed_sigma=args.fixed_sigma)
else:
    agent_mi = AgentPicker(env, model, args.device_np, args.num_context, custom_reward=None, pick_dist=None,
                           mean_action=False, render=False, running_state=None, fixed_sigma=args.fixed_sigma)


def train_mi(datasets, epochs=args.epochs_per_iter):
    print('Policy training')
    data_loader = DataLoader(datasets, batch_size=1, shuffle=True)
    model_trainer.train_rl_loo(data_loader, args.epochs_per_iter, early_stopping=None)


avg_rewards_mi = [0]
tot_steps_mi = [0]


def main_loop():
    improved_context_list_mi = sample_initial_context_normal(env)
    for i_iter in range(args.max_iter_num):

        # define context set
        if len(replay_memory_mi) == 0 or not args.rm_as_context:
            context_list_np = improved_context_list_mi
        else:
            context_list_np = replay_memory_mi.data

        # collect samples
        batch_mi, log_mi = agent_mi.collect_episodes(context_list_np, args.num_req_steps, args.num_ensembles)

        # compute discounted rewards
        disc_rew_mi = discounted_rewards(batch_mi.memory, args.gamma)
        iter_dataset_mi = BaseDataset(batch_mi.memory, disc_rew_mi, args.device_np, args.dtype,  max_len=max_episode_len)

        # estimate advantages
        if args.value_net:
            state_list = [ep['states'][:ep['real_len']] for ep in iter_dataset_mi]
            advantages_mi, returns = critic_estimate(value_net, state_list, disc_rew_mi, args)
            update_critic(value_net, torch.cat(state_list, dim=0), torch.cat(returns, dim=0))
        else:
            advantages_mi = estimate_v_a(iter_dataset_mi, disc_rew_mi, value_replay_memory, model, args)
            value_replay_memory.add(iter_dataset_mi)

        # update step
        improved_context_list_mi = improvement_step_all(iter_dataset_mi, advantages_mi, args.max_kl_mi, args)

        # training
        replay_memory_mi.add(iter_dataset_mi)
        train_mi(replay_memory_mi)

        # prints & plots
        tot_steps_mi.append(tot_steps_mi[-1] + log_mi['num_steps'])
        avg_rewards_mi.append(log_mi['avg_reward'])
        if i_iter % args.log_interval == 0:
            print(i_iter)
            print('mi: \tR_min {:.2f} \tR_max {:.2f} \tR_avg {:.2f}'.format(log_mi['min_reward'], log_mi['max_reward'], log_mi['avg_reward']))
        print('new sigma', args.fixed_sigma)
        store_avg_rewards(tot_steps_mi[-1], log_mi['avg_reward'], mi_file.replace(str(args.seed)+'.csv', 'avg'+str(args.seed)+'.csv'))
        if i_iter % args.plot_every == 0:
            plot_rewards_history(tot_steps_mi,avg_rewards_mi)
        if tot_steps_mi[-1] > args.tot_steps:
            break
    """clean up gpu memory"""
    torch.cuda.empty_cache()



def plot_rewards_history(steps, rews):
    fig_rew, ax_rew = plt.subplots(1, 1)

    ax_rew.plot(steps[1:], rews[1:], c='g', label='Mean Interpolator')
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