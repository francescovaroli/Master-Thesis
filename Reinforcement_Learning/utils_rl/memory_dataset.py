from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


def get_random_context(context_list, num_context):

        all_x = torch.cat([ep[0][:, :ep[-1], :] for ep in context_list], dim=-2)
        all_y = torch.cat([ep[1][:, :ep[-1], :] for ep in context_list], dim=-2)
        num_tot_context = all_x.shape[-2]
        num_context = min(num_tot_context, num_context)
        locations = np.random.choice(np.arange(num_tot_context),
                                     size=num_context,
                                     replace=False)
        x_context = all_x[:, locations, :]
        y_context = all_y[:, locations, :]
        return x_context, y_context


def get_close_context(index, target, context_list, dist, num_tot_context=1000):
    """
    Select subset of the context set either by index or distance
    """
    if dist is not None:
        x1_target = target[0, 0, 0].cpu()
        x2_target = target[0, 0, 1].cpu()
        x1_close = []
        x2_close = []
        x3_close = []
        for episode in context_list:
            real_len = episode[-1]
            for x1, x2, x3 in zip(episode[0][0,:real_len,0], episode[0][0,:real_len,1], episode[1][0,:real_len,0]):
                if np.linalg.norm([x1_target-x1.cpu(), x2_target-x2.cpu()]) < dist:
                    x1_close.append(x1)
                    x2_close.append(x2)
                    x3_close.append(x3)
        if len(x1_close) == 0:
            x1_close.append(episode[0][0,index,0])
            x2_close.append(episode[0][0,index,1])
            x3_close.append(episode[1][0,index,0])
        closer_context = [torch.tensor([x1_close, x2_close]).transpose(1,0).unsqueeze(0),
                          torch.tensor(x3_close).view(1,-1,1)]
        return closer_context
    else:
        num_all_context = 0
        for e in context_list:
            num_all_context += e[-1]
        num_tot_context = min(num_tot_context, num_all_context)
        context_per_ep = max(1, num_tot_context // len(context_list))
        start = max(0, index - context_per_ep // 2)
        end = min(start + context_per_ep, context_list[0][2])
        if end - start < context_per_ep:
            start = max(0, end - context_per_ep)
        chosen_context = [context_list[0][0][..., start:end, :].view(1, end-start, -1),
                          context_list[0][1][..., start:end, :].view(1, end-start, -1)]  # need to use ... and view because tensor shape varies (1xNxD in training, NxD in inference
        for ep in context_list[1:]:
            start = max(0, index - context_per_ep // 2)
            end = min(start + context_per_ep, ep[2])
            if end - start < context_per_ep:
                start = max(0, end - context_per_ep)
            chosen_context[0] = torch.cat([chosen_context[0], ep[0][..., start:end, :].view(1, end-start, -1)], dim=-2)
            chosen_context[1] = torch.cat([chosen_context[1], ep[1][..., start:end, :].view(1, end-start, -1)], dim=-2)
        return chosen_context



def merge_context(context_points_list, perc=1.):
    '''Transforms a list of episodes' context points (padded to max_len)
     into a vector with all points unpadded'''
    all_x_context, all_y_context, real_len = context_points_list[0]
    all_x_context = all_x_context[...,:real_len,:].view(1, real_len, -1) # need to use ... and view because tensor shape varies (1xNxD in training, NxD in inference
    all_y_context = all_y_context[...,:real_len,:].view(1, real_len, -1)
    for episode_contexts in context_points_list[1:]:
        x_context, y_context, real_len = episode_contexts
        all_x_context = torch.cat((all_x_context, x_context[...,:real_len,:].view(1, real_len, -1)), dim=-2)
        all_y_context = torch.cat((all_y_context, y_context[...,:real_len,:].view(1, real_len, -1)), dim=-2)
    return all_x_context, all_y_context

def merge_padded_list(points_list, max_lens):
    '''Transforms a list of episodes' attributes (padded to max_len)
     into a vector with all points unpadded'''
    first_len = max_lens[0]
    all_points = points_list[0][:first_len,:]
    for points, real_len in zip(points_list[1:], max_lens[1:]):
        all_points = torch.cat((all_points, points[:first_len,:]), dim=0)
    return all_points

def merge_padded_lists(*args, max_lens=[]):
    '''Transforms a list of episodes' attributes (padded to max_len)
     into a vector with all points unpadded'''
    first_len = max_lens[0]
    merged = []
    for points_list in args:
        all_points = points_list[0][:first_len,:]
        for points, real_len in zip(points_list[1:], max_lens[1:]):
            all_points = torch.cat((all_points, points[:real_len,:]), dim=0)
        merged.append(all_points)
    return merged



class BaseDataset(Dataset):
    """
    Basic dataset to covert episodes of list of transactions to episodes of list of states, actions, means, stddev, rewards
    """
    def __init__(self, memory_list, disc_rew, device, dtype, max_len=None):
        self.data = []
        self.keys = ['states', 'actions', 'means', 'stddevs', 'rewards', 'covariances', 'discounted_rewards', 'real_len']

        state_dim = len(memory_list[0][0].state)
        action_dim = (memory_list[0][0].action).size
        unpadded_data = []
        for e, episode in enumerate(memory_list):
            trajectory_states = []
            trajectory_actions = []
            trajectory_means = []
            trajectory_stddevs = []
            trajectory_rewards = []
            trajectory_covariances = []
            trajectory_discounted_rewards = []
            len_traj = len(episode)
            for t, transition in enumerate(episode):
                trajectory_states.append(transition.state)
                trajectory_actions.append(transition.action)
                trajectory_means.append(transition.mean)
                trajectory_stddevs.append(transition.stddev)
                trajectory_rewards.append(transition.reward)
                trajectory_covariances.append(transition.covariance)
                trajectory_discounted_rewards.append(disc_rew[e][t])

            unpadded_data.append([torch.tensor(trajectory_states).to(dtype).to(device).view(-1, state_dim),
                                  torch.tensor(trajectory_actions).to(dtype).to(device).view(-1, action_dim),
                                  torch.tensor(trajectory_means).to(dtype).to(device).view(-1, action_dim),
                                  torch.tensor(trajectory_stddevs).to(dtype).to(device).view(-1, action_dim),
                                  torch.tensor(trajectory_rewards).to(dtype).to(device).view(-1, 1),
                                  torch.stack(trajectory_covariances).to(dtype).to(device),
                                  torch.tensor(trajectory_discounted_rewards).to(dtype).to(device).view(-1, 1),
                                  len_traj])
        _, _, _, _, _, _, _, lengths = zip(*unpadded_data)
        if max_len is None:
            max_len = max(lengths)

        for unpad_traj in unpadded_data:  # pad the trajectory to max_len
            pad_traj = {}
            for i, unpad_values in enumerate(unpad_traj[:-1]):
                if unpad_values.dim() == 3:
                    pad_v = torch.zeros([max_len, unpad_values.shape[-2], unpad_values.shape[-1]]).to(dtype).to(device)
                else:
                    pad_v = torch.zeros([max_len, unpad_values.shape[-1]]).to(dtype).to(device)
                pad_v[:unpad_traj[-1], ...] = unpad_traj[i]
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

    def __init__(self, memory_size, use_mean=True):
        self.data = []
        self.max_size = memory_size
        self.use_mean = use_mean

    def add(self, dataset):
        add_len = len(dataset)
        my_len = len(self.data)
        excess = my_len + add_len - self.max_size
        if excess > 0:
            del self.data[0:excess]

        for episode in dataset:
            #if self.use_mean:
            self.data.append([episode['states'], episode['new_means'], episode['real_len']])
            #else:
            #    self.data.append([episode['states'], episode['new_actions'], episode['real_len']])

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
        """Saves a list of transitions."""
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
        len_traj = 0
        trajectory_states = []
        trajectory_means = []
        rew = []
        for transition in memory_list:
            trajectory_states.append(transition.state)
            trajectory_means.append(transition.mean.view(1, action_dim))
            rew.append(transition.reward)
            len_traj += 1
            if transition.mask == 0:
                unpadded_data.append([torch.tensor(trajectory_states).to(dtype).to(device),
                                      torch.cat(trajectory_means, dim=0).to(dtype).to(device),
                                      len_traj])
                self.rewards.append(rew)
                len_traj = 0
                trajectory_states = []
                trajectory_means = []
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


class MemoryDatasetTRPO_old(Dataset):
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