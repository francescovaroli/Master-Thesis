import os
import sys
import csv
import numpy as np
from matplotlib import pyplot as plt

folder_path = '/media/francesco/Irene/Francesco/Master Thesis/scratch/Swimmer-v2_NP:True_MI:True_10epi_fixSTD:0,5_' \
              '0,999gamma__NP_10,6rm_60,60epo_32z_128h_0,1kl_attention:True_128a_MI_10rm_60epo_50z_64h_0,5kl_uniform/mi/'
chunk_size = 5000
num_seeds = 3
data = []
for i in range(num_seeds):
    file_path = folder_path + str(i) +'.csv'
    data.append(np.genfromtxt(file_path))

min_len = min(len(l) for l in data)
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

plt.plot(np.arange(1, len(avg_rews)+1)*chunk_size, avg_rews)
plt.show()