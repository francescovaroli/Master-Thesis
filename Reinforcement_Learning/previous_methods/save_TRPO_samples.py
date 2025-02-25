import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils_rl import *
from trpo_model.mlp_policy import Policy
from core.mlp_critic import Value
from trpo_model.mlp_policy_disc import DiscretePolicy
from trpo_model.trpo import trpo_step
from core.common import estimate_advantages
from core.agent import Agent


parser = argparse.ArgumentParser(description='PyTorch TRPO example')
parser.add_argument('--env-name', default="MountainCarContinuous-v0", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                    help='damping (default: 1e-2)')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=13, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=5000, metavar='N',
                    help='minimal batch size per TRPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=1000, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
use_running_state = False
"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
if use_running_state:
    running_state = ZFilter((state_dim,), clip=5)  # running list of states that allows to access precise mean and std
else:
    running_state = None
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""define actor and critic"""
if args.model_path is None:
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
    value_net = Value(state_dim)
else:
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
policy_net.to(device)
value_net.to(device)

"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, render=args.render, num_threads=args.num_threads)


def update_params(batch):
    # (3)
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)  # estimate value function of each state with NN

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """perform TRPO update"""
    trpo_step(policy_net, value_net, states, actions, returns, advantages, args.max_kl, args.damping, args.l2_reg)

run_id = 'deb'
directory_path = '/home/francesco/PycharmProjects/MasterThesis/value learning/' + run_id

def plot_policy(net, info):
    # Axes3D import has side effects, it enables using projection='3d' in add_subplot
    import matplotlib.pyplot as plt
    fig = plt.figure()
    bounds_high = env.observation_space.high
    bounds_low = env.observation_space.low
    x1 = np.linspace(bounds_low[0], bounds_high[0], 100)
    x2 = np.linspace(bounds_low[1], bounds_high[1], 100)
    X1, X2 = np.meshgrid(x1, x2)
    zs = []
    for _x1 in x1:
        for _x2 in x2:
            state = tensor([_x1, _x2], device=device).unsqueeze(0)
            zs.append(net(state)[0][0].item())
    Z = (np.array(zs)).reshape(X1.shape).transpose()

    ax = plt.axes(projection='3d')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    if info[2] == 'policies':
        ax.set_zlabel('Acceleration')
        ax.set_zlim(env.action_space.low, env.action_space.high)
        ax.plot_surface(X1, X2, Z, cmap='viridis',  vmin=-1., vmax=1.)  #

    else:
        ax.set_zlabel('Value')
        ax.set_zlim(-50, 100)
        ax.plot_surface(X1, X2, Z, cmap='viridis', vmin=-50., vmax=100.)  # ,

    name = 'TRPO {} iter: {} avg_rew: {}'.format(info[2], info[0], int(info[1]))
    ax.set_title(name, pad=20)
    fig.savefig(directory_path+'/'+info[2]+'/'+name)
    plt.close(fig)
    # plt.show()

def create_directories(directory_path):
    os.mkdir(directory_path)
    os.mkdir(directory_path+'/policies/')
    os.mkdir(directory_path+'/values/')
    #os.mkdir(directory_path+'/episodes/')


def create_memory_file(i_iter):
    idd = str(i_iter) + '^iter_' + args.env_name
    memory_file = directory_path + '/episodes/' + idd
    os.mknod(memory_file)
    return memory_file

def main_loop():
    for i_iter in range(args.max_iter_num):
        # (1)
        #memory_file = create_memory_file(i_iter)

        # generate multiple trajectories that reach the minimum batch_size
        batch, log, memory = agent.collect_samples(args.min_batch_size)  # batch of at least batch_siz transitions from multiple
        print(log['num_steps'], log['num_episodes'])                     # episodes (separated by mask=0). Stored in Memory

        # TRPO step
        t0 = time.time()
        update_params(batch)  # estimate advantages from samples and update policy by TRPO step
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))
            plot_policy(policy_net, (i_iter, log['avg_reward'], 'policies'))
            plot_policy(value_net, (i_iter, log['avg_reward'], 'values'))


        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net)
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_trpo.p'.format(args.env_name)), 'wb'))
            to_device(device, policy_net, value_net)

            #  Saving samples from this iteration as file
            with open(memory_file, 'wb') as file_m:
                pickle.dump(memory, file_m, protocol=pickle.HIGHEST_PROTOCOL)


create_directories(directory_path)
main_loop()
