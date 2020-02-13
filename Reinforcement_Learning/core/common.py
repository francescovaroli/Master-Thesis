import torch
from utils_rl import to_device
from utils_rl.memory_dataset import rewards_from_batch


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
    #rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
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
    :param rewards: list of list, each of the inner is a trajectory of transitions
    :return: list of list
    '''

    rewards = rewards_from_batch(batch)

    disc_rewards = []

    for traj_rewards in rewards:
        T = len(traj_rewards) - 1
        R = torch.zeros(T + 1)
        R[-1] = traj_rewards[-1]
        for t in range(T-1, -1, -1):
            # R[t] = R[t + 1] + gamma ** (T - t) * traj_rewards[t]
            R[t] = gamma * R[t + 1] + traj_rewards[t]
        disc_rewards.append(R)
    return disc_rewards



#rewards = [[1]*10, [1]*5, [1]*2]
#print(discounted_rewards(rewards, 0.9))
    # traj_disc_rew = [0]
    #        num_steps = len(traj_rewards)
    #    for t in reversed(range(num_steps)):
    #        curr_disc_rew = gamma * (traj_disc_rew[0] + traj_rewards[t])
#        traj_disc_rew.insert(0, curr_disc_rew)