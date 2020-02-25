import csv

def rewards_from_batch(batch):
    rewards = []
    for episode in batch:
        episode_rewards = []
        for transition in episode:
            episode_rewards.append(transition.reward)
        rewards.append(episode_rewards)
    return rewards


def store_avg_rewards(step, avg_rew, rewards_file):
    with open(rewards_file, mode='a+') as write_file:
        reward_writer = csv.writer(write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        reward_writer.writerow([step, avg_rew.item()])

def store_rewards(batch, rewards_file):
    ep_rewards = rewards_from_batch(batch)
    with open(rewards_file, mode='a+') as write_file:
        reward_writer = csv.writer(write_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for rewards in ep_rewards:
            for rew in rewards:
                reward_writer.writerow([rew.item()])

