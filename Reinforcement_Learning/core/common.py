import torch
from utils_rl import to_device


def estimate_advantages(rewards, masks, values, gamma, tau, device):
    """
    Estimate normalized advantages and Q-values of all state-action pairs sampled from a batch
    :param rewards:
    :param masks:
    :param values:
    :param gamma:
    :param tau:
    :param device:
    :return:
    """
    # (4)
    rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]  # at the end of every episode m=0 so we're
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]  # computing from there backwards each time
        #                           why are we adding prev adv?
        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / advantages.std()

    advantages, returns = to_device(device, advantages, returns)
    return advantages, returns


def discounted_rewards(batch, gamma):
    '''
    :param rewards: list of list, each of the inner is a trajectory of rewards
    :return: list of list
    '''

    rewards = []
    for episode in batch:
        episode_rewards = []
        for transition in episode:
            episode_rewards.append(transition.reward)
        rewards.append(episode_rewards)

    disc_rewards = []
    # de-mean
    #for e in range(len(rewards)):
    #    avg = sum(rewards[e]) / len(rewards[e])
    #    rewards[e] = [float(r) - avg for r in rewards[e]]

    for traj_rewards in rewards:
        traj_disc_rew = [0]
        num_steps = len(traj_rewards)
        for t in reversed(range(num_steps)):
            curr_disc_rew = gamma * (traj_disc_rew[0] + traj_rewards[t])
            traj_disc_rew.insert(0, curr_disc_rew)
        disc_rewards.append(traj_disc_rew[:-1])

    return disc_rewards



rewards = [[1]*10, [1]*5, [1]*2]
#print(discounted_rewards(rewards, 0.9))