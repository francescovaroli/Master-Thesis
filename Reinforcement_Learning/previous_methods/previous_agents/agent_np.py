import multiprocessing
from utils_rl.memory_dataset import Memory
from utils_rl.torch import *
import math
import time
from collections import namedtuple
from torch.distributions import Normal

Transition = namedtuple('Transition', ('state', 'action', 'next_state',
                                       'reward', 'mean', 'stddev', 'disc_rew'))

def collect_samples(pid, queue, env, policy, custom_reward,
                    mean_action, render, running_state, min_batch_size, context_points):
    # (2)
    torch.randn(pid)
    log = dict()
    memory = Memory()  # every time we collect a batch he memory is re-initialized
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0
    ## should take only the real unpaddded
    x_context = context_points[0]
    y_context = context_points[1]
    _, z_dist = policy.sample_z(x_context, y_context)
    while num_steps < min_batch_size:  # collecting samples from episodes until we at least a batch
        state = env.reset()            # (maybe more since we stop when episode ends)
        if running_state is not None:
            state = running_state(state)
        reward_episode = 0
        episode = []
        z_sample = z_dist.sample()
        for t in range(10000):  # in gym.env there's already an upper bound to the number of steps
            state_var = tensor(state).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                mean, stddev = policy.xz_to_y(state_var, z_sample)
                action_distribution = Normal(mean, stddev)
                if mean_action:
                    action = mean  # use mean value
                    mean, stddev = policy.xz_to_y(state_var, z_dist.mean)
                else:
                    action = action_distribution.sample().squeeze(0).squeeze(0)  # sample from normal distribution

            #action = int(action) if policy.is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            reward_episode += reward
            if running_state is not None:  # running list of normalized states allowing to access precise mean and std
                next_state = running_state(next_state)
            if custom_reward is not None:  # by default is None, unless given when init Agent
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            episode.append(Transition(state, action.numpy(), next_state, reward, mean.numpy(), stddev.numpy(), None))

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

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log

def compute_stats(batch):
    s = 0
    ma = -100
    mi = 100
    l = 0
    for ep in batch:
        l += len(ep)
        for tr in ep:
            action = tr.action
            s += action
            ma = max(ma, action)
            mi = min(mi, action)
    avg = s / l
    return avg, ma, mi


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env, policy, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_threads=1):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads

    def collect_samples(self, min_batch_size, context=None):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env, self.policy, self.custom_reward, self.mean_action,
                           False, self.running_state, thread_batch_size, context)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env, self.policy, self.custom_reward, self.mean_action,
                                      self.render, self.running_state, thread_batch_size, context)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.memory  # not sampling but giving all memory
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        to_device(self.device, self.policy)
        t_end = time.time()
        mean_a, max_a, min_a = compute_stats(batch)
        log['sample_time'] = t_end - t_start
        log['action_mean'] = mean_a
        log['action_min'] = min_a
        log['action_max'] = max_a
        return batch, log, memory
