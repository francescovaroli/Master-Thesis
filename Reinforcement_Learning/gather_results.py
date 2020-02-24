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


folder_path = '/media/francesco/Irene/Francesco/Master Thesis/scratch/MI_hopper'
all_folders = list_folders(folder_path)
num_folders = len(all_folders)
alpha = 2.5/num_folders
max_len = 1000000
chunk_size = 5000
num_seeds = 3

fig_rew, ax_rew = plt.subplots(1, 1)
ax_rew.set_xlabel('number of steps')
ax_rew.set_ylabel('average reward')
title = 'Average Reward History'
ax_rew.set_title(title)

for s, subfolder_path in enumerate(all_folders[1:]):
    print('...', s)
    data = []
    for i in range(num_seeds):
        file_path = subfolder_path + '/{}.csv'.format(i)
        data.append(np.genfromtxt(file_path))

    min_len = min(min(len(l) for l in data), max_len)
    data_avg = np.zeros(min_len)
    for d in data:
        data_avg += d[:min_len]/num_seeds

    start = 0
    avg_rews = []
    while start < min_len:
        end = min(start + chunk_size, min_len)
        chunk = data_avg[start:end]
        avg = sum(chunk)/len(chunk)
        avg_rews.append(avg)
        start = end
    label = 'Mean Kernel Interpolation'
    ax_rew.plot(np.arange(1, len(avg_rews)+1)*chunk_size, avg_rews, alpha=alpha, c='g', label=label)

    if s == 0: plt.legend()
plt.grid()
fig_rew.savefig(folder_path+title)
plt.close(fig_rew)
