from torch.utils.data import Dataset, DataLoader
import torch
import random


class BaseDataset(Dataset):
    """
    Basic dataset to covert episodes of list of transactions to episodes of list of states, actions, means, stddev, rewards
    """
    def __init__(self, memory_list, disc_rew,device, dtype, max_len=None):
        self.data = []
        self.keys = ['states', 'actions', 'means', 'stddevs', 'rewards', 'discounted_rewards', 'real_len']

        state_dim = len(memory_list[0][0].state)
        action_dim = len(memory_list[0][0].action)
        unpadded_data = []
        for e, episode in enumerate(memory_list):
            trajectory_states = []
            trajectory_actions = []
            trajectory_means = []
            trajectory_stddevs = []
            trajectory_rewards = []
            trajectory_discounted_rewards = []
            len_traj = len(episode)
            for t, transition in enumerate(episode):
                trajectory_states.append(transition.state)
                trajectory_actions.append(transition.action)
                trajectory_means.append(transition.mean)
                trajectory_stddevs.append(transition.stddev)
                trajectory_rewards.append(transition.reward)
                trajectory_discounted_rewards.append(disc_rew[e][t])

            unpadded_data.append([torch.tensor(trajectory_states).to(dtype).to(device).view(-1,state_dim),
                                  torch.tensor(trajectory_actions).to(dtype).to(device).view(-1,action_dim),
                                  torch.tensor(trajectory_means).to(dtype).to(device).view(-1,action_dim),
                                  torch.tensor(trajectory_stddevs).to(dtype).to(device).view(-1,action_dim),
                                  torch.tensor(trajectory_rewards).to(dtype).to(device).view(-1,1),
                                  torch.tensor(trajectory_discounted_rewards).to(dtype).to(device).view(-1,1),
                                  len_traj])
        _, _, _, _, _, _, lengths = zip(*unpadded_data)
        if max_len is None:
            max_len = max(lengths)

        for unpad_traj in unpadded_data:
            pad_traj = {}
            for i, unpad_values in enumerate(unpad_traj[:-1]):
                pad_v = torch.zeros([max_len, len(unpad_values[0])]).to(dtype).to(device)
                pad_v[:unpad_traj[-1], :] = unpad_traj[i]
                pad_traj[self.keys[i]] = pad_v
            pad_traj['real_len'] = unpad_traj[-1]
            self.data.append(pad_traj)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ValueReplay(Dataset):

    def __init__(self, memory_size):
        self.data = []
        self.max_size = memory_size


    def add(self, complete_dataset):
        add_len = len(complete_dataset)
        my_len = len(self.data)
        excess = my_len + add_len - self.max_size
        if excess > 0:
            del self.data[0:excess]
        for episode in complete_dataset:
            self.data.append([episode['states'], episode['discounted_rewards'], episode['real_len']])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ReplayMemoryDataset(Dataset):
    """
    Fixed size Memory to aggregate datasets
    memory_size: max number of datasets (iterations)
    """

    def __init__(self, memory_size):
        self.data = []
        self.max_size = memory_size

    def add(self, dataset):
        add_len = len(dataset)
        my_len = len(self.data)
        excess = my_len + add_len - self.max_size
        if excess > 0:
            del self.data[0:excess]
        self.data += dataset.data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MemoryDatasetNP(Dataset):
    """
    Dataset to store memory
    The memory is a list of transitions (tuples) from many trajectory collected during one iteration.
    We transform it to a dataset where the data is a list of trajectories (each traj is a list of 2 tensor, x and y)
    """
    def __init__(self, memory_list, new_means, device, dtype, max_len=None):
        self.data = []
        self.rewards = []

        state_dim = len(memory_list[0][0].state)
        action_dim = len(memory_list[0][0].action)

        unpadded_data = []
        idx = 0
        for episode in memory_list:
            trajectory_states = []
            trajectory_actions = []
            trajectory_rewards = []
            len_traj = len(episode)
            for transition in episode:
                trajectory_states.append(transition.state)
                trajectory_actions.append(new_means[0][idx].numpy())
                trajectory_rewards.append(transition.reward)
                idx += 1
            unpadded_data.append([torch.tensor(trajectory_states).to(dtype).to(device),
                                  torch.tensor(trajectory_actions).to(dtype).to(device),
                                  len_traj])
            self.rewards.append(trajectory_rewards)
        xs, ys, lengths = zip(*unpadded_data)
        if max_len is None:
            max_len = max(lengths)

        for unpad_traj in unpadded_data:
            pad_x = torch.zeros([max_len, state_dim]).to(dtype).to(device)
            pad_y = torch.zeros([max_len, action_dim]).to(dtype).to(device)
            pad_x[:unpad_traj[2], :] = unpad_traj[0]
            pad_y[:unpad_traj[2], :] = unpad_traj[1]
            self.data.append([pad_x, pad_y, unpad_traj[2]])

    def __getitem__(self, index):
        return self.data[index]

    def get_rewards(self, index):
        return self.rewards[index]

    def __len__(self):
        return len(self.data)



class Memory(object):
    '''
    list of episodes
    '''
    def __init__(self):
        self.memory = []

    def push(self, episode):
        """Saves a transition."""
        self.memory.append(episode)

    def set_disc_rew(self, rew_list):
        if len(self.memory) != len(rew_list):
            raise ValueError('Rewards len different from num of episodes')
        else:
            for e, trajectory in enumerate(self.memory):
                if len(trajectory) != len(rew_list[e]):
                    raise ValueError('Num of rewards in trajectory {} different from episode len'.format(e))
                else:
                    for t, transition in enumerate(trajectory):
                        self.memory[e][t] = transition._replace(disc_rew=rew_list[e][t])


    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)


class MemoryDatasetTRPO(Dataset):
    """
    Dataset to store memory
    The memory is a list of transitions (tuples) from many trajectory collected during one iteration.
    We transform it to a dataset where the data is a list of trajectories (each traj is a list of 2 tensor, x and y)
    """
    def __init__(self, memory_list, device, dtype, max_len=None):
        self.data = []
        self.rewards = []

        state_dim = len(memory_list[0].state)
        action_dim = len(memory_list[0].action)

        unpadded_data = []
        for transition in memory_list:
            len_traj = 0
            trajectory_states = []
            trajectory_actions = []
            rew = []
            trajectory_states.append(transition.state)
            trajectory_actions.append(transition.action)
            rew.append(transition.reward)
            len_traj += 1
            if transition.mask == 0:
                unpadded_data.append([torch.tensor(trajectory_states).to(dtype).to(device),
                                      torch.tensor(trajectory_actions).to(dtype).to(device),
                                      len_traj])
                self.rewards.append(rew)

        xs, ys, lengths = zip(*unpadded_data)  # make it dependent on an input parameter 'max_lenght', if None this
        if max_len is None:
            max_len = max(lengths)

        for unpad_traj in unpadded_data:
            pad_x = torch.zeros([max_len, state_dim]).to(dtype).to(device)
            pad_y = torch.zeros([max_len, action_dim]).to(dtype).to(device)
            pad_x[:unpad_traj[2], :] = unpad_traj[0]
            pad_y[:unpad_traj[2], :] = unpad_traj[1]
            self.data.append([pad_x, pad_y, unpad_traj[2]])

    def __getitem__(self, index):
        return self.data[index]

    def get_rewards(self, index):
        return self.rewards[index]

    def __len__(self):
        return len(self.data)


class ReplayMemoryDatasetTRPO(Dataset):
    """
    Fixed size Memory to aggregate datasets
    memory_size: max number of datasets (iterations)
    """
    def __init__(self, memory_size):
        self.data = []
        self.rewards = []
        self.max_size = memory_size

    def add(self, dataset):
        add_len = len(dataset)
        my_len = len(self.data)
        excess = my_len + add_len - self.max_size
        if excess > 0:
            del self.data[0:excess]
            del self.rewards[0:excess]
        self.data += dataset.data
        self.rewards += dataset.rewards

    def __getitem__(self, index):
        return self.data[index]

    def get_rewards(self, index):
        return self.rewards[index]

    def __len__(self):
        return len(self.data)


class ValueDataset(Dataset):
    def __init__(self, value_training_set):
        self.data = value_training_set

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)