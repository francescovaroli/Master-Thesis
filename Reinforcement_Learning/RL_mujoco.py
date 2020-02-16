import argparse
import gym
import os
import sys
import time
from random import randint
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils_rl import *
from core.common import discounted_rewards
from core.agent_ensembles_all_context import Agent_all_ctxt
from neural_process import NeuralProcess
from training_leave_one_out import NeuralProcessTrainerLoo
from training_module_RL import NeuralProcessTrainerRL
from MeanInterpolatorModel import MeanInterpolator, MITrainer
import csv
from multihead_attention_np import *
from torch.distributions import Normal
from models.mlp_policy import Policy
from models.mlp_critic import Value
from core.trpo import trpo_step
from core.common import estimate_advantages
from core.agent import Agent


torch.set_default_tensor_type(torch.DoubleTensor)
if torch.cuda.is_available() and False:
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    device = torch.device("cpu")
print('device: ', device)

parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="HalfCheetah-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--use-trpo', default=True, help='trpo')
parser.add_argument('--use-np', default=True, help='np')
parser.add_argument('--use-mi', default=True, help='mi')


parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl-trpo', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
parser.add_argument('--num-threads', type=int, default=5, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--min-batch-size', type=int, default=2994, metavar='N',
                    help='minimal batch size per TRPO update (default: 2048)')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')

parser.add_argument('--z-mi-dim', type=int, default=50, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--h-mi-dim', type=int, default=356, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--scaling', default='uniform', metavar='N',
                    help='feature extractor scaling')
parser.add_argument("--lr_nn", type=float, default=1e-4,
                    help='plot every n iter')

parser.add_argument('--use-running-state', default=False,
                    help='store running mean and variance instead of states and actions')
parser.add_argument('--max-kl-np', type=float, default=0.06, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--max-kl-mi', type=float, default=0.06, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--num-ensembles', type=int, default=3, metavar='N',
                    help='episode to collect per iteration')
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')

parser.add_argument('--use-mean', default=True, metavar='N',
                    help='train & condit on improved means/actions'),
parser.add_argument('--fixed-sigma', default=0.35, metavar='N',
                    help='sigma of the policy')
parser.add_argument('--epochs-per-iter', type=int, default=60, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--replay-memory-size', type=int, default=20, metavar='G',
                    help='size of training set in episodes ')
parser.add_argument('--z-dim', type=int, default=32, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--r-dim', type=int, default=128, metavar='N',
                    help='dimension of representation space in np')
parser.add_argument('--h-dim', type=int, default=128, metavar='N',
                    help='dimension of hidden layers in np')
parser.add_argument('--np-batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--early-stopping', type=int, default=-1000, metavar='N',
                    help='stop training training when avg_loss reaches it')

parser.add_argument('--v-epochs-per-iter', type=int, default=60, metavar='G',
                    help='training epochs of NP')
parser.add_argument('--v-replay-memory-size', type=int, default=4, metavar='G',
                    help='size of training set in episodes')
parser.add_argument('--v-z-dim', type=int, default=128, metavar='N',
                    help='dimension of latent variable in np')
parser.add_argument('--v-r-dim', type=int, default=128, metavar='N',
                    help='dimension of representation space in np')
parser.add_argument('--a-dim', type=int, default=128, metavar='N',
                    help='dimension of representation space in np')
parser.add_argument('--v-np-batch-size', type=int, default=1, metavar='N',
                    help='batch size for np training')
parser.add_argument('--v-early-stopping', type=int, default=-1000, metavar='N',
                    help='stop training training when avg_loss reaches it')

parser.add_argument('--directory-path', default='/home/francesco/PycharmProjects/MasterThesis/mujoco learning results/',
                    help='path to plots folder')
parser.add_argument('--tot-steps', default=1000000,
                    help='total steps in the run')


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


np_spec = '_NP_{},{}rm_{},{}epo_{}z_{}h_{}kl_attention:{}_{}a'.format(args.replay_memory_size, args.v_replay_memory_size,
                                                                args.epochs_per_iter,args.v_epochs_per_iter, args.z_dim,
                                                                args.h_dim, args.max_kl_np, args.use_attentive_np, args.a_dim)

mi_spec = '_MI_{}rm_{}epo_{}z_{}h_{}kl_{}'.format(args.replay_memory_size, args.epochs_per_iter, args.z_mi_dim,
                                          args.h_mi_dim,args.max_kl_mi, args.scaling)

run_id = '/{}_NP:{}_MI:{}_{}epi_fixSTD:{}_{}gamma_'.format(args.env_name, args.use_np, args.use_mi,
                                                                   args.num_ensembles, args.fixed_sigma,
                                                                    args.gamma) + np_spec + mi_spec
run_id = run_id.replace('.', ',')
args.directory_path += run_id

trpo_file = args.directory_path + '/trpo/{}.csv'.format(args.seed)
np_file = args.directory_path + '/np/{}.csv'.format(args.seed)
mi_file = args.directory_path + '/mi/{}.csv'.format(args.seed)

max_episode_len = 1000
#torch.set_default_dtype(args.dtype)

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0

args.fixed_sigma = args.fixed_sigma * torch.ones(action_dim)
"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

policy_net = Policy(state_dim, action_dim, log_std=args.log_std).to(device)
value_net = Value(state_dim).to(device)


agent_trpo = Agent(env, policy_net, device, running_state=None, render=args.render, num_threads=1)

def update_params_trpo(batch):
    # (3)
    states = torch.from_numpy(np.stack(batch.state)).to(args.dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(args.dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(args.dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(args.dtype).to(device)
    with torch.no_grad():
        values = value_net(states)  # estimate value function of each state with NN

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """perform TRPO update"""
    trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl_trpo, args.damping, args.l2_reg)


model = MeanInterpolator(state_dim, args.h_mi_dim, args.z_mi_dim, scaling=args.scaling).to(device).double()

optimizer_mi = torch.optim.Adam([
    {'params': model.feature_extractor.parameters(), 'lr': args.lr_nn},
    {'params': model.interpolator.parameters(), 'lr': args.lr_nn}])
# train
model_trainer = MITrainer(device, model, optimizer_mi, args, print_freq=50)
replay_memory_mi = ReplayMemoryDataset(args.replay_memory_size, use_mean=args.use_mean)


'''create neural process'''
if args.use_attentive_np:
    policy_np = AttentiveNeuralProcess(state_dim, action_dim, args.r_dim, args.z_dim, args.h_dim,
                                                       args.a_dim, use_self_att=False).to(args.device_np)
else:
    policy_np = NeuralProcess(state_dim, action_dim, args.r_dim, args.z_dim, args.h_dim).to(args.device_np)

optimizer = torch.optim.Adam(policy_np.parameters(), lr=3e-4)
np_trainer = NeuralProcessTrainerLoo(args.device_np, policy_np, optimizer,
                                    print_freq=50)

if args.v_use_attentive_np:
    value_np = AttentiveNeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_r_dim,
                                      args.a_dim, use_self_att=False).to(args.device_np)
else:
    value_np = NeuralProcess(state_dim, 1, args.v_r_dim, args.v_z_dim, args.v_h_dim).to(args.device_np)
value_optimizer = torch.optim.Adam(value_np.parameters(), lr=3e-4)
value_np_trainer = NeuralProcessTrainerLoo(args.device_np, value_np, value_optimizer,
                                          print_freq=50)
value_np.training = False
"""create replay memory"""
replay_memory = ReplayMemoryDataset(args.replay_memory_size, use_mean=args.use_mean)
value_replay_memory = ValueReplay(args.v_replay_memory_size)

"""create agent"""
agent_np = Agent_all_ctxt(env, policy_np, args.device_np, running_state=None, render=args.render,
              attention=args.use_attentive_np, fixed_sigma=args.fixed_sigma)
agent_mi = Agent_all_ctxt(env, model, args.device_np, running_state=None, render=args.render,
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

def train_mi(datasets, epochs=args.epochs_per_iter):
    print('Policy training')
    data_loader = DataLoader(datasets, batch_size=args.np_batch_size, shuffle=True)
    model_trainer.train_rl_loo(data_loader, args.epochs_per_iter, early_stopping=None)


def estimate_eta_3(actions, means, advantages, sigmas, covariances, is_np):
    """Compute learning step from all the samples of previous iteration"""
    n,  _, d = actions.size()
    stddev = args.fixed_sigma
    if is_np:
        eps = tensor(args.max_kl_np).to(args.dtype)
    else:
        eps = tensor(args.max_kl_mi).to(args.dtype)

    T = tensor(actions.shape[0]).to(args.dtype)
    if d > 1:  # a, m, s: Nx1xD, disc_r: Nx1, cov: NxDxD
        diff_vector = (actions - means) / sigmas   # Nx1xD
        squared_normalized = (diff_vector.matmul(covariances)).matmul(diff_vector.transpose(1,2)).view(-1)  # (Nx1xD)(NxDxD)(NxDx1) -> (Nx1xD)(NxDx1) -> Nx1x1
        denominator = (0.5 * (advantages ** 2).matmul(squared_normalized))  # (1xN)(Nx1) -> 1

    else:
        iter_sum = 0
        for action, mean, disc_reward, sigma in zip(actions, means, advantages, sigmas):
            if stddev is None:
                stddev = sigma
            iter_sum += ((disc_reward ** 2) * (action - mean) ** 2) / (2 * (stddev ** 4))
        denominator = iter_sum.to(args.dtype)
    return torch.sqrt((T * eps) / denominator)


def improvement_step_all(complete_dataset, estimated_adv, is_np):
    """Perform improvement step using same eta for all episodes"""
    all_improved_context = []
    with torch.no_grad():
        all_states, all_means, all_stdv, all_actions, all_covariances = merge_padded_lists([episode['states'] for episode in complete_dataset],
                                                                [episode['means'] for episode in complete_dataset],
                                                                [episode['stddevs'] for episode in complete_dataset],
                                                                [episode['actions'] for episode in complete_dataset],
                                                                [episode['covariances'] for episode in complete_dataset],
                                                                 max_lens=[episode['real_len'] for episode in complete_dataset])
        all_advantages = torch.cat(estimated_adv, dim=0).view(-1)
        eta = estimate_eta_3(all_actions.unsqueeze(1), all_means.unsqueeze(1), all_advantages, all_stdv.unsqueeze(1),
                             all_covariances, is_np)
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
                # new_padded_actions[i, :] = new_action
                new_padded_means[i, :] = new_mean
                i += 1
            episode['new_means'] = new_padded_means
            # episode['new_actions'] = new_padded_actions
            #if args.use_mean:
            all_improved_context.append([episode['states'].unsqueeze(0), new_padded_means.unsqueeze(0), real_len])
            #else:
            #    all_improved_context.append([episode['states'].unsqueeze(0), new_padded_actions.unsqueeze(0), real_len])

    return all_improved_context



def estimate_v_a(complete_dataset, disc_rew):
    ep_rewards = disc_rew
    ep_states = [ep['states'] for ep in complete_dataset]
    real_lens = [ep['real_len'] for ep in complete_dataset]
    estimated_advantages = []
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
    return estimated_advantages


def estimate_v_a_mi(complete_dataset, disc_rew):
    ep_rewards = disc_rew
    ep_states = [ep['states'] for ep in complete_dataset]
    real_lens = [ep['real_len'] for ep in complete_dataset]
    estimated_advantages = []
    for i in range(len(ep_states)):
        context_list = []
        j = 0
        for states, rewards, real_len in zip(ep_states, ep_rewards, real_lens):
            if j != i:
                context_list.append([states.unsqueeze(0), rewards.view(1,-1,1), real_len])
            else:
                s_target = states[:real_len, :].unsqueeze(0)
                r_target = rewards.view(1,-1,1)
            j += 1
        s_context, r_context = merge_context(context_list)
        with torch.no_grad():
            # values = model(s_context, r_context, s_target)
            values = torch.cat([model(s_context, r_context, s_target[:, :real_lens[i]//2, :]),
                                model(s_context, r_context, s_target[:, real_lens[i]//2:, :])], dim=-2)
        advantages = r_target - values
        estimated_advantages.append(advantages.squeeze(0))
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



avg_rewards_np = [0]
tot_steps_np = [0]
avg_rewards_trpo = [0]
tot_steps_trpo = [0]
avg_rewards_mi = [0]
tot_steps_mi = [0]

def store_rewards_trpo(batch, rewards_file):
    rewards = [tr.reward for tr in batch]
    with open(rewards_file, mode='a+') as employee_file:
        reward_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for rew in rewards:
            reward_writer.writerow([rew.item()])


def store_rewards(batch, rewards_file):
    ep_rewards = rewards_from_batch(batch)
    with open(rewards_file, mode='a+') as employee_file:
        reward_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for rewards in ep_rewards:
            for rew in rewards:
                reward_writer.writerow([rew.item()])

def main_loop():
    colors = []
    num_episodes = args.num_ensembles
    for i in range(num_episodes):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
        improved_context_list_np = sample_initial_context_normal(args.num_ensembles)
        improved_context_list_mi = improved_context_list_np
        if args.use_np:
            if initial_training:
                train_on_initial(improved_context_list_np)
    for i_iter in range(args.max_iter_num):
        if args.use_trpo and tot_steps_trpo[-1] < args.tot_steps:
            batch_trpo, log, memory_trpo = agent_trpo.collect_samples(args.min_batch_size)  # batch of batch_size transitions from multiple
            store_rewards_trpo(memory_trpo.memory, trpo_file)
            update_params_trpo(batch_trpo)  # generate multiple trajectories that reach the minimum batch_size
            tot_steps_trpo.append(tot_steps_trpo[-1] + log['num_steps'])
            avg_rewards_trpo.append(log['avg_reward'])
            print('trpo avg actions: ', log['action_mean'])
        if args.use_np and tot_steps_np[-1] < args.tot_steps:
            # generate multiple trajectories that reach the minimum batch_size
            policy_np.training = False
            batch_np, log_np = agent_np.collect_episodes(improved_context_list_np)  # batch of batch_size transitions from multiple
            store_rewards(batch_np.memory, np_file)
            disc_rew_np = discounted_rewards(batch_np.memory, args.gamma)
            complete_dataset_np = BaseDataset(batch_np.memory, disc_rew_np, args.device_np, args.dtype,  max_len=max_episode_len)
            print('np avg actions: ', log_np['action_mean'])
            advantages_np = estimate_v_a(complete_dataset_np, disc_rew_np)

            improved_context_list_np = improvement_step_all(complete_dataset_np, advantages_np, is_np=True)
            # training
            value_replay_memory.add(complete_dataset_np)
            train_value_np(value_replay_memory)

            tn0 = time.time()
            replay_memory.add(complete_dataset_np)
            train_np(replay_memory)
            tn1 = time.time()
            tot_steps_np.append(tot_steps_np[-1] + log_np['num_steps'])
            avg_rewards_np.append(log_np['avg_reward'])

        if args.use_mi and tot_steps_mi[-1] < args.tot_steps:
            # generate multiple trajectories that reach the minimum batch_size
            batch_mi, log_mi = agent_mi.collect_episodes(improved_context_list_mi)  # batch of batch_size transitions from multiple
            store_rewards(batch_mi.memory, mi_file)
            #print(log['num_steps'], log['num_episodes'])                # episodes (separated by mask=0). Stored in Memory
            print('mi avg actions: ', log_mi['action_mean'])

            disc_rew_mi = discounted_rewards(batch_mi.memory, args.gamma)
            complete_dataset_mi = BaseDataset(batch_mi.memory, disc_rew_mi, args.device_np, args.dtype,  max_len=max_episode_len)
            advantages_mi = estimate_v_a_mi(complete_dataset_mi, disc_rew_mi)

            t0 = time.time()
            improved_context_list_mi = improvement_step_all(complete_dataset_mi, advantages_mi, is_np=False)
            t1 = time.time()

            # create training set
            tn0 = time.time()
            replay_memory_mi.add(complete_dataset_mi)
            train_mi(replay_memory_mi)
            tn1 = time.time()
            tot_steps_mi.append(tot_steps_mi[-1] + log_mi['num_steps'])

            avg_rewards_mi.append(log_mi['avg_reward'].item())
        if i_iter % args.log_interval == 0:
            print(i_iter)
            if args.use_trpo:
                print('trpo: \tR_min {:.2f} \tR_max {:.2f} \tR_avg {:.2f}'.format(log['min_reward'], log['max_reward'], log['avg_reward']))
            if args.use_np:
                print('np: \tR_min {:.2f} \tR_max {:.2f} \tR_avg {:.2f}'.format(log_np['min_reward'], log_np['max_reward'], log_np['avg_reward']))
            if args.use_mi:
                print('mi: \tR_min {:.2f} \tR_max {:.2f} \tR_avg {:.2f}'.format(log_mi['min_reward'], log_mi['max_reward'], log_mi['avg_reward']))

        if i_iter % args.plot_every == 0:
            plot_rewards_history([tot_steps_trpo, tot_steps_np, tot_steps_mi],
                                 [avg_rewards_trpo, avg_rewards_np, avg_rewards_mi])

    """clean up gpu memory"""
    torch.cuda.empty_cache()



def plot_rewards_history(steps, rews):
    fig_rew, ax_rew = plt.subplots(1, 1)
    labels = ['TRPO', 'NP', 'MI']
    colors = ['r', 'b', 'g']
    for i in range(len(steps)):
        ax_rew.plot(steps[i], rews[i], c=colors[i], label=labels[i])
    ax_rew.set_xlabel('number of steps')
    ax_rew.set_ylabel('average reward')
    ax_rew.set_title('Average Reward History')
    plt.legend()
    plt.grid()
    fig_rew.savefig(args.directory_path + run_id.replace('.', ','))
    plt.close(fig_rew)


def create_directories(directory_path):

    os.mkdir(directory_path)
    os.mkdir(directory_path + '/trpo')
    os.mkdir(directory_path + '/np')
    os.mkdir(directory_path + '/mi')

try:
    create_directories(args.directory_path)
except FileExistsError:
    pass

main_loop()