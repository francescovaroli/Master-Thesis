import gym

env = 'CartPole-v0'
num_episodes = 10
num_steps = 100

env = gym.make(env)
for ep in range(num_episodes):
    observation = env.reset()
    for t in range(num_steps):
        env.render()
        action = env.action_space.sample()
        observation, rew, done, info = env.step(action)
        print(observation, rew, action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()
