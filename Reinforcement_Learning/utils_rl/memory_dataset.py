from torch.utils.data import Dataset, DataLoader
import torch


class ReplayMemoryDataset(Dataset):
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
        self.data += dataset.data

    def __getitem__(self, index):
        return self.data[index]

    def get_rewards(self, index):
        return self.rewards[index]

    def __len__(self):
        return len(self.data)

class MemoryDataset(Dataset):
    """
    Dataset to store memory
    The memory is a list of transactions (tuples) from many trajectory collected during one iteration.
    We transform it to a dataset where the data is a list of trajectories (each traj is a list of 2 tensor, x and y)
    """
    def __init__(self, memory_list, device, dtype, max_len=None):
        self.data = []
        self.rewards = []

        state_dim = len(memory_list[0].state)
        action_dim = len(memory_list[0].action)

        unpadded_data = []
        trajectory_states = []
        trajectory_actions = []
        rew = []
        len_traj = 0
        for transition in memory_list:
            trajectory_states.append(transition.state)
            trajectory_actions.append(transition.action)
            rew.append(transition.reward)
            len_traj += 1
            if transition.mask == 0:
                unpadded_data.append([torch.tensor(trajectory_states).to(dtype).to(device),
                                  torch.tensor(trajectory_actions).to(dtype).to(device), len_traj])
                self.rewards.append(rew)
                len_traj = 0
                trajectory_states = []
                trajectory_actions = []
                rew = []
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
