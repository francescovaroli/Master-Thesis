import multiprocessing
from utils_rl.memory_dataset import Memory, merge_context, get_close_context
from utils_rl.torch_ut import *
import math
import time
from collections import namedtuple
from torch.distributions import Normal, MultivariateNormal

Transition = namedtuple('Transition', ('state', 'action', 'next_state',
                                       'reward', 'mean', 'stddev', 'disc_rew', 'covariance'))

def collect_samples(pid, env, policy, num_req_steps, num_req_episodes, num_context, render,
                    running_state, context_points_list, pick_dist, fixed_sigma):

    torch.randn(pid)
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0
    action_sum = zeros(context_points_list[0][1].shape[-1])

    # merge all episodes in RM in a single set
    all_x = torch.cat([ep[0][:ep[-1], :] for ep in context_points_list], dim=-2)
    all_y = torch.cat([ep[1][:ep[-1], :] for ep in context_points_list], dim=-2)
    num_tot_context = all_x.shape[-2]

    if num_tot_context < num_context:  # no need to select a subset
        pick = False
        all_x_context, all_y_context = [all_x.view(1, num_tot_context, -1), all_y.view(1, num_tot_context, -1)]
    else:
        pick = True

    with torch.no_grad():
        while num_steps < num_req_steps or num_episodes < num_req_episodes:
            # print('ep: ', ep)
            episode = []
            reward_episode = 0

            state = env.reset()
            if running_state is not None:
                state = running_state(state)
            t_ep = time.time()
            for t in range(10000):
                state_var = tensor(state).unsqueeze(0).unsqueeze(0)
                if pick:
                    all_x_context, all_y_context = get_close_context(t, state_var, context_points_list, pick_dist, num_tot_context=num_context)
                if policy.id == 'DKL':
                    policy.set_train_data(all_x_context.squeeze(0), all_y_context.squeeze(0).squeeze(-1), strict=False)
                    pi = policy(state_var)
                    mean = pi.mean
                    stddev = pi.stddev
                elif policy.id == 'MI':
                    mean = policy(all_x_context, all_y_context, state_var)
                    stddev = fixed_sigma
                else:
                    pi = policy(all_x_context, all_y_context, state_var)
                    mean = pi.mean
                    stddev = pi.stddev

                if fixed_sigma is not None:
                    sigma = fixed_sigma
                else:
                    sigma = stddev.view(-1)
                cov = torch.diag(sigma ** 2)

                action_distribution = MultivariateNormal(mean, cov)
                action = action_distribution.sample().view(-1)  # sample from normal distribution

                next_state, reward, done, _ = env.step(action.cpu().numpy())
                reward_episode += reward
                if running_state is not None:  # running list of normalized states allowing to access precise mean and std
                    next_state = running_state(next_state)

                episode.append(Transition(state, action.cpu().numpy(), next_state, reward, mean.cpu().numpy(),
                                          sigma.cpu().numpy(), None, cov))
                action_sum += action
                if render:
                    env.render()
                if done:
                    memory.push(episode)
                    break

                state = next_state
            # log stats
            num_steps += (t + 1)
            num_episodes += 1
            total_reward += reward_episode
            min_reward = min(min_reward, reward_episode)
            max_reward = max(max_reward, reward_episode)
    print('tot episodes: ', num_episodes)
    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    try:
        log['avg_reward'] = total_reward.item() / num_episodes
    except AttributeError:
        log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    log['action_mean'] = action_sum / num_steps

    return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])

    return log


class AgentPicker:
    """
    Agent collecting samples for IMeL.
    The policy are sampled by the NP/MKI selecting a subset of the context set.
    """
    def __init__(self, env, policy, device, num_context, pick_dist=None,
                 mean_action=False, render=False, running_state=None, fixed_sigma=None):
        self.env = env
        self.policy = policy
        self.device = device
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.pick_dist = pick_dist
        self.fixed_sigma = fixed_sigma
        self.num_context = num_context

    def collect_episodes(self, context_list, num_steps, num_ep, render=False):
        t_start = time.time()
        #to_device(torch.device('cpu'), self.policy)

        memory, log = collect_samples(0, self.env, self.policy, num_steps, num_ep, self.num_context,
                                      self.render, self.running_state, context_list, self.pick_dist, self.fixed_sigma)

        batch = memory.memory
        #to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        return memory, log
