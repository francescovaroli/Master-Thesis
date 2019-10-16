import numpy as np
import gym
from gym import wrappers

class BinaryActionLinearPolicy(object):
    def __init__(self, theta):
        self.w = theta[:-1]
        self.b = theta[-1]

    def act(self, ob):
        y = ob.dot(self.w) + self.b
        a = int(y < 0)
        return a


def cem(f, theta_mean, batch_size, n_iter, elite_fraction, initial_std=1.0):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function

    f: a function mapping from vector -> scalar
    theta_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_fraction: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size * elite_fraction))
    theta_std = np.ones_like(theta_mean)  # theta_dim =  obs_dim + 1  (w + bias)

    for _ in range(n_iter):
        thetas = np.array([theta_mean + d_theta for d_theta in theta_std[None, :]
                           *np.random.randn(batch_size, theta_mean.size)])  # batch_size*theta_dim
        ys = np.array([f(theta) for theta in thetas])  # batch_size*1  (y is the reward)
        elite_inds = ys.argsort()[::-1][:n_elite]  # n_elite*1
        elite_thetas = thetas[elite_inds]  # n_elite*theta_dim
        theta_mean = elite_thetas.mean(axis=0)  # 1*theta_dim
        theta_std = elite_thetas.std(axis=0)  # 1*theta_dim
        yield {'ys': ys, 'theta_mean': theta_mean, 'y_mean': ys.mean()}

def do_rollout(agent, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        action = agent.act(ob)
        (ob, rew, done, _info) = env.step(action)
        total_rew += rew
        if render and t%2==0:
            env.render()
        if done:
            break
    return total_rew, t+1

if __name__ == '__main__':

    env = gym.make("CartPole-v0")
    env.seed(0)
    np.random.seed(0)

    outdir = '/tmp/cem-agent-results'
    env = wrappers.Monitor(env, outdir, force=True)

    num_steps = 200
    params = dict(n_iter=10, batch_size=25, elite_fraction=0.2)

    def noisy_evaluation(theta):
        agent = BinaryActionLinearPolicy(theta)
        rew, T = do_rollout(agent, env, num_steps)
        return rew

    # Training
    for (i, iterdata) in enumerate(cem(noisy_evaluation, np.zeros(env.observation_space.shape[0]+1), **params)):
        print('Iteration %2i. Episode mean reward: %7.3f'%(i, iterdata['y_mean']))
        agent = BinaryActionLinearPolicy(iterdata['theta_mean'])
        do_rollout(agent, env, 200, render=True)
    env.close()