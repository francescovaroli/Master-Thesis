import matplotlib.pyplot as plt
import numpy as np
from neural_process import NeuralProcess
from training_module import NeuralProcessTrainer
import torch
from torch.utils.data import DataLoader

device = torch.device('cpu')
torch.set_default_tensor_type(torch.DoubleTensor)

neuralprocess = NeuralProcess(1, 1, 32, 32, 32)
optimizer = torch.optim.Adam(neuralprocess.parameters(), lr=0.0005)
np_trainer = NeuralProcessTrainer(device, neuralprocess, optimizer,
                                  num_context_range=(50, 60),
                                  num_extra_target_range=(90, 101),
                                  print_freq=50000)
neuralprocess.training = True
eta = 0.05
sigma = 0.1
T = 151

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
np_trainer.train(data_loader, 8000, early_stopping=-100)
neuralprocess.training = False
pred = neuralprocess(x_tor.unsqueeze(0), y_tor.unsqueeze(0), x_tor.unsqueeze(0))

plt.xlabel('State')
plt.ylabel('Action')
plt.plot(x, pred.mean.view(-1).detach(), c='b', label='New policy mean')
plt.legend()
cb = plt.colorbar(a)
cb.set_label('Advantages')
plt.show()
