import os
import sys
import csv
import numpy as np
from matplotlib import pyplot as plt

def list_folders(dir):
    subfold = []
    for folder in os.walk(dir):
        subfold.append(folder[0])
    return subfold


folder_path = '/media/francesco/Irene/Francesco/Master Thesis/scratch/np_hop_09/'
all_folders = list_folders(folder_path)
num_folders = len(all_folders)
alpha = 5/num_folders
max_len = 1000000
chunk_size = 50000
num_seeds = 3

fig_rew, ax_rew = plt.subplots(1, 1)
ax_rew.set_xlabel('number of steps')
ax_rew.set_ylabel('average reward')
title = 'Average Reward History'
ax_rew.set_title(title)
if 'NP' in all_folders[1]:
    label = 'Attentive Neural Process'
    color = 'b'
elif 'MI' in all_folders[1]:
    label = 'Mean Kearnel Interpolation'
    color = 'g'
elif 'MLP' in all_folders[1]:
    label = 'Multi Layer Perceptron'
    color = 'magenta'

for s, subfolder_path in enumerate(all_folders[1:]):
    if 'no loo' in subfolder_path:
        continue
    step_data = []
    rew_data = []
    lens = []
    for i in range(num_seeds):
        file_path = subfolder_path + '/avg{}.csv'.format(i)
        data = np.genfromtxt(file_path, delimiter=',')
        step_data.extend(data[:, 0])
        rew_data.extend(data[:, 1])
        lens.append(data[-1, 0])
    min_len = min(min(lens), max_len)

    start = 0
    avg_rews = []
    steps = []
    while start < min_len:
        end = min(start + chunk_size, min_len)
        indexes = [i for i in range(len(step_data)) if start < step_data[i] <= end]
        chunk = np.take(rew_data, indexes)
        start = end
        if len(chunk) == 0:
            print('missing ', start, end)
            continue
        avg = sum(chunk)/len(chunk)
        avg_rews.append(avg)
        steps.append(end)
    if '40rm' in subfolder_path:
        color = 'r'
    elif '100rm' in subfolder_path:
        color = 'g'
    else: color='b'
    ax_rew.plot(np.arange(1, len(avg_rews)+1)*chunk_size, avg_rews, alpha=alpha, c=color, label=label)
    print('...', s)

    if s == 0: plt.legend()
plt.grid()
fig_rew.savefig(folder_path+title)
plt.close(fig_rew)
