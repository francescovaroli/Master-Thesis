from utils_rl.memory_dataset import merge_padded_lists, merge_context
from RL_results.store_results import rewards_from_batch
import scipy.optimize
from utils_rl.torch_ut import *


def estimate_advantages(rewards, masks, values, gamma, tau, device):
    """
    Estimate normalized advantages for TRPO algorithm
    """

    #rewards, masks, values = to_device(torch.device('cpu'), rewards, masks, values)
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1)
    advantages = tensor_type(rewards.size(0), 1)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]  # at the end of every episode m=0 so we're
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]  # computing from there backwards each time
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



def estimate_eta_3(actions, means, advantages, sigmas, covariances, eps, args):
    """Compute learning step from all the samples of previous iteration"""
    n,  _, d = actions.size()
    stddev = args.fixed_sigma

    T = torch.tensor(n).to(args.dtype)
    if d>1:  # a, m, s: Nx1xD, disc_r: Nx1, cov: NxDxD
        diff_vector = (actions - means) / sigmas**2   # Nx1xD
        squared_normalized = (diff_vector.matmul(torch.inverse(covariances))).matmul(diff_vector.transpose(1,2)).view(-1)  # (Nx1xD)(NxDxD)(NxDx1) -> (Nx1xD)(NxDx1) -> Nx1x1
        denominator = (0.5 * (advantages ** 2).matmul(squared_normalized))  # (1xN)(Nx1) -> 1
        if torch.isnan(torch.sqrt((T * eps) / denominator)):
            print('nan in eta')
    else:
        iter_sum = 0
        for action, mean, disc_reward, sigma in zip(actions, means, advantages, sigmas):
            if stddev is None:
                stddev = sigma
            iter_sum += ((disc_reward ** 2) * (action - mean) ** 2) / (2 * (stddev ** 6))
        denominator = iter_sum.to(args.dtype)
    return torch.sqrt((T * eps) / denominator)


def improvement_step_all(complete_dataset, estimated_adv, eps, args):
    """Perform improvement step using same eta for all episodes"""
    all_improved_context = []
    new_sigma_list = []
    all_m = []
    all_new_m = []
    with torch.no_grad():
        all_states, all_means, all_stdv, all_actions, all_covariances = merge_padded_lists([episode['states'] for episode in complete_dataset],
                                                                [episode['means'] for episode in complete_dataset],
                                                                [episode['stddevs'] for episode in complete_dataset],
                                                                [episode['actions'] for episode in complete_dataset],
                                                                [episode['covariances'] for episode in complete_dataset],
                                                                 max_lens=[episode['real_len'] for episode in complete_dataset])
        all_advantages = torch.cat(estimated_adv, dim=0).view(-1)
        eta = estimate_eta_3(all_actions.unsqueeze(1), all_means.unsqueeze(1), all_advantages, all_stdv.unsqueeze(1),
                             all_covariances, eps, args)
        for episode, episode_adv in zip(complete_dataset, estimated_adv):
            real_len = episode['real_len']
            states = episode['states'][:real_len]
            actions = episode['actions'][:real_len]
            means = episode['means'][:real_len]
            stddevs = episode['stddevs'][:real_len]
            new_padded_means = torch.zeros_like(episode['means'])

            i = 0
            for state, action, mean, advantage, stddev in zip(states, actions, means, episode_adv, stddevs):
                if args.fixed_sigma is None:
                    sigma = stddev
                else:
                    sigma = args.fixed_sigma
                new_mean = mean + eta * advantage * ((action - mean) / sigma**2)
                new_padded_means[i, :] = new_mean
                if args.learn_sigma:
                    new_sigma_list.append(eta * advantage * (((action - mean)**2 - sigma**2)/sigma**3))
                all_m.append(mean)
                all_new_m.append(new_mean)
                i += 1
            episode['new_means'] = new_padded_means
            all_improved_context.append([episode['states'].unsqueeze(0), new_padded_means.unsqueeze(0), real_len])
    # print('avg kl diff: ', (0.5*((((torch.stack(all_new_m, dim=0)-torch.stack(all_m, dim=0))**2)/(all_stdv**2)).mean(0).sum())))
    if args.learn_sigma and args.fixed_sigma is not None:
        new_sigma = torch.stack(new_sigma_list, dim=0).mean(dim=0)
        args.fixed_sigma += new_sigma.view(args.fixed_sigma.shape)


    return all_improved_context


def compute_gae(rewards, values, gamma, tau):
    """
    Compute gae advantages for IMeL algorithm
    """
    tensor_type = type(values)
    deltas = tensor_type(values.size(0), 1)
    advantages = tensor_type(values.size(0), 1)
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage
        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    return advantages

def estimate_v_a(iter_dataset, disc_rew, value_replay_memory, model, args):
    '''
    Function to estimate the advantage by predicting the baseline V(s) = V^NN(s)
    and the advantage A(s,t) = Q(a, t)- V(t)
    '''
    ep_rewards = disc_rew
    ep_states = [ep['states'] for ep in iter_dataset]
    real_lens = [ep['real_len'] for ep in iter_dataset]
    estimated_advantages = []
    estimated_values = []
    if not len(value_replay_memory.data) == 0:
        s_context, r_context = merge_context(value_replay_memory.data)
        for states, rewards, real_len in zip(ep_states, ep_rewards, real_lens):
            s_target = states[:real_len, :].unsqueeze(0)
            r_target = rewards.view(1, -1, 1)
            with torch.no_grad():
                values = model(s_context, r_context, s_target)
            if 'NP' in model.id:
                values = values.mean
            advantages = r_target - values
            estimated_values.append(values.squeeze(0))
            estimated_advantages.append(advantages.squeeze(0))
    else:
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
                values = model(s_context, r_context, s_target)
            if 'NP' in model.id:
                values = values.mean
            advantages = r_target - values
            estimated_values.append(values.squeeze(0))
            estimated_advantages.append(advantages.squeeze(0))
    if args.gae:
        gae_advantages = []
        for rew, val in zip(ep_rewards, estimated_values):
            gae_adv = compute_gae(rew, val, args.gamma, args.tau)
            gae_advantages.append(gae_adv)
            returns = val + gae_adv
        return gae_advantages
    return estimated_advantages


def sample_initial_context_normal(env, initial_num=20, init_sigma=0.1):
    """
    Sample states and means for initial context C^0
    """
    initial_episodes = []
    max_episode_len = env._max_episode_steps
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    state_sigma = []
    for l, h in zip(env.observation_space.low, env.observation_space.high):
        s_l = max(l, -1000)
        s_h = min(h, 1000)
        state_sigma.append(torch.tensor((s_h - s_l) / 4))
    state_sigma = torch.stack(state_sigma)

    for e in range(initial_num):
        states = torch.zeros([1, max_episode_len, state_dim])
        for i in range(max_episode_len):
            states[:, i, :] = torch.normal(torch.zeros(state_dim), state_sigma)  #  torch.from_numpy(env.observation_space.sample())

        dims = [1, max_episode_len, action_dim]
        actions_init = torch.normal(torch.zeros(dims), init_sigma*torch.ones(dims))
        initial_episodes.append([states, actions_init, max_episode_len])

    return initial_episodes


def critic_estimate(value_net, states_list, rewards_list, args):
    """
    compute advantages
    """
    adv_list = []
    ret_list = []
    for states, rewards in zip(states_list, rewards_list):
        with torch.no_grad():
            values = value_net(states)
        advantages = compute_gae(rewards, values, args.gamma, args.tau)
        returns = values + advantages
        adv_list.append(advantages)
        ret_list.append(returns)
    return adv_list, ret_list


def update_critic(value_net, states, returns, l2_reg=1e-3):
    def get_value_loss(flat_params):
        """
        compute the loss for the value network comparing estimated values with empirical ones
        and optimizes the network parameters
        :param flat_params:
        :return:
        """
        set_flat_params_to(value_net, tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()  # MeanSquaredError

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return value_loss.item(), get_flat_grad_from(value_net.parameters()).cpu().numpy()

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                            get_flat_params_from(value_net).detach().cpu().numpy(),
                                                            maxiter=25)
    set_flat_params_to(value_net, tensor(flat_params))
    return value_net
