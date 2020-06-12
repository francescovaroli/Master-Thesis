import numpy as np
from neural_process import NeuralProcess
from training_module import NeuralProcessTrainer
import torch
from torch.utils.data import DataLoader

def plot_pca_proj(dataset, advantages, model):
    import time
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    with torch.no_grad():

        pca = PCA(n_components=1)
        real_len =dataset.data[0]['real_len']
        states = dataset.data[0]['states'][:real_len, :].detach().cpu()
        means = dataset.data[0]['means'][:real_len, :].detach().cpu()
        actions = dataset.data[0]['actions'][:real_len, :].detach().cpu()
        improved = dataset.data[0]['new_means'][:real_len, :].detach().cpu()
        adv = advantages[0].detach().cpu()
        pca_states = pca.fit_transform(states)
        model.training = False
        pred = model(states.cuda().unsqueeze(0), improved.cuda().unsqueeze(0), states.cuda().unsqueeze(0))
        if model.id == 'MI':
            new_means = pred.view(-1).detach().cpu()
        else:
            new_means = pred.mean.view(-1).detach().cpu()
        pca_states, means, actions, improved, adv, new_means = [torch.tensor(t, device='cpu') for t in zip(*sorted(zip(pca_states, means, actions, improved, adv, new_means)))]

        fig = plt.figure(figsize=(12, 8))

        plt.plot(pca_states, means, c='r', label='Previous policy mean')

        plt.scatter(pca_states, actions, cmap='viridis', c=adv, label='Sampled actions')
        a = plt.scatter(pca_states, improved, cmap='viridis', marker='+', c=adv, label='Updated means')


        plt.xlabel('PCA state projection')
        #plt.title('MountainCarContinuous-v0')
        plt.title('CartPole-v0')
        plt.ylabel('Action')
        #plt.ylim(-1, 2)
        plt.plot(pca_states, new_means, c='b', label='New policy mean')
        plt.legend()
        cb = plt.colorbar(a)
        cb.set_label('Advantage') # $Q_{\pi}(a, s)$
        plt.grid()
        fig.savefig('/home/francesco/PycharmProjects/MasterThesis/plots/update_step/'+time.ctime())
        plt.close(fig)


def plot_sim1D():
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.DoubleTensor)

    neuralprocess = NeuralProcess(1, 1, 32, 32, 32)
    optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=0.0005)
    np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                      num_context_range=(50, 60),
                                      num_extra_target_range=(90, 101),
                                      print_freq=50000)
    neuralprocess.training = True
    eta = 5.
    sigma = 0.1
    T = 151
    epochs = 1000

    x = np.linspace(-1, 1, T)
    mu = x**2 - 0.5
    act = np.random.normal(0, sigma, T)
    y = mu+act
    rew = np.linspace(10, -10, T) + np.random.normal(0, 0.5, T) + act*15
    y_new = mu + rew*eta*(act/sigma)

    fig = plt.figure(figsize=(9, 6))

    plt.plot(x, mu, c='r', label='Previous policy mean')
    plt.scatter(x, y, cmap='viridis', c=rew, label='Sampled actions')
    a = plt.scatter(x, y_new, cmap='viridis', marker='+', c=rew, label='Updated means')

    x_tor = torch.tensor(x).unsqueeze(-1)
    y_tor = torch.tensor(y_new).unsqueeze(-1)
    dataset = [[x_tor, y_tor]]
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    np_trainer.train(data_loader, epochs, early_stopping=-100)
    neuralprocess.training = False
    pred = neuralprocess(x_tor.unsqueeze(0), y_tor.unsqueeze(0), x_tor.unsqueeze(0))

    plt.xlabel('State')
    plt.ylabel('Action')
    plt.ylim(-1, 2)
    plt.plot(x, pred.mean.view(-1).detach(), c='b', label='New policy mean')
    plt.legend()
    cb = plt.colorbar(a)
    cb.set_label('Advantages')
    plt.show()
