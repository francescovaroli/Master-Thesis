import multiprocessing
from utils_rl.memory_dataset import Memory, merge_context
from utils_rl.torch import *
import math
import time
from collections import namedtuple
from torch.distributions import Normal
import gpytorch

Transition = namedtuple('Transition', ('state', 'action', 'next_state',
                                       'reward', 'mean', 'stddev', 'disc_rew', 'covariance'))

def collect_samples(pid, env, policy, num_req_steps, num_req_episodes, mean_action, render,
                    running_state, context_points_list, attention, fixed_sigma):

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

    with torch.no_grad():
        all_x_context, all_y_context = merge_context(context_points_list)  # merge episodes in one context set
        #  compute step-independent values
        if policy.id == 'DKL':
            policy.set_train_data(inputs=all_x_context.squeeze(0), targets=all_y_context.view(-1), strict=False)
        elif policy.id in 'ANP':
            #  compute context representation and latent variable
            if attention:
                encoder_input, keys = policy.xy_to_a.get_input_key(all_x_context, all_y_context)
            else:
                r_context = policy.xy_to_r(all_x_context, all_y_context)
            _, z_dist = policy.sample_z(all_x_context, all_y_context)

        while num_steps < num_req_steps or num_episodes < num_req_episodes:
            #print('episode', num_episodes)

            episode = []
            reward_episode = 0
            if policy.id in 'ANP':
                z_sample = z_dist.sample()
                if not attention:
                    rep = torch.cat([z_sample, r_context], dim=-1)

            state = env.reset()
            if running_state is not None:
                state = running_state(state)
            t_ep = time.time()
            for t in range(10000):
                state_var = tensor(state).unsqueeze(0).unsqueeze(0)
                if policy.id == 'DKL':
                    with gpytorch.settings.use_toeplitz(True), gpytorch.settings.fast_pred_var():
                        pi = policy(state_var)
                    mean = pi.mean
                    stddev = pi.stddev

                elif policy.id == 'MI':
                    mean = policy(all_x_context, all_y_context, state_var)
                    stddev = fixed_sigma
                else:  #  NPs and ANPs
                    if attention:
                        a_repr = policy.xy_to_a.get_repr(encoder_input, keys, state_var)
                        representation = torch.cat([z_sample, a_repr.squeeze(0)], dim=-1)
                        mean, stddev = policy.xz_to_y(state_var, representation)
                    else:
                        mean, stddev = policy.xrep_to_y(state_var, rep)

                if fixed_sigma is not None:
                    sigma = fixed_sigma      # use sigma learnt by update step
                else:
                    sigma = stddev.view(-1)  # use predicted sigma (NPs)

                action_distribution = Normal(mean, sigma)

                if mean_action:
                    action = mean.view(-1)  # use mean value
                    mean_rep = torch.cat([z_dist.mean, r_context], dim=-1)
                    mean, stddev = policy.xrep_to_y(state_var, mean_rep)
                    mean_s, _ = policy.xrep_to_y(state_var, torch.cat([z_dist.mean + z_dist.stddev, r_context], dim=-1))
                    sigma = torch.abs(mean_s - mean)
                else:
                    action = action_distribution.sample().view(-1)   # sample from normal distribution
                cov = torch.diag(sigma.view(-1)**2)

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


class Agent_all_ctxt:

    def __init__(self, env, policy, device, attention=False,
                 mean_action=False, render=False, running_state=None, fixed_sigma=None):
        self.env = env
        self.policy = policy
        self.device = device
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.attention = attention
        self.fixed_sigma = fixed_sigma

    def collect_episodes(self, context_list, num_steps, num_ep):
        t_start = time.time()
        # to_device(torch.device('cpu'), self.policy)

        memory, log = collect_samples(0, self.env, self.policy, num_steps, num_ep, self.mean_action,
                                      self.render, self.running_state, context_list, self.attention, self.fixed_sigma)

        batch = memory.memory
        # to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start

        return memory, log
