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

def separate_subfolders(folders, keys):
    subf = []
    for key in keys:
        key_subf = []
        for folder in folders:
            if key in folder:
                key_subf.append(folder)
        subf.append(key_subf)
    return subf
for sub in ['ant']:
    folder_path = '/media/francesco/Irene/Francesco/Master Thesis/scratch/np_want/'+sub

    keys = ['RM: 20', 'RM: 150', 'MaI', '$\sigma_{min}^{NP}$: 0,1 ' ,'$\sigma_{min}^{NP}$: 0,15 ','$\sigma_{min}^{NP}$: 0,2 ', '$\sigma_{min}^{NP}$: 0,25 ' ]

    all_folders = list_folders(folder_path)
    num_folders = len(all_folders)
    alpha = 0.6
    max_len = 1000000
    chunk_size = 20000
    num_seeds = 3
    first = 0
    fig_rew, ax_rew = plt.subplots(1, 1)
    ax_rew.set_xlabel('Number of steps')
    ax_rew.set_ylabel('Average reward')
    title = 'Reward History Percentile ' + all_folders[2].split('-')[0].split('/')[-1]
    colors = ['b', 'r', 'g', 'c' ,'g', 'y']
    add_label = [' (no context)', ' (context)']
    #ax_rew.set_title(title)
    for e, subfolder in enumerate(separate_subfolders(all_folders[1:], keys)):
        if subfolder == [] or 'nop' in subfolder:
            continue
        if 'NP' in subfolder[0]:
            label = 'Attentive Neural Process'
            color = 'b'
        elif 'MI' in subfolder[0]:
            label = 'Mean Kearnel Interpolation'
            color = 'g'
        elif 'MLP' in subfolder[0]:
            label = 'Multi Layer Perceptron'
            color = 'magenta'
        elif 'TRPO' in subfolder[0]:
            label = 'TRPO'
            color = 'r'

        rew_param = []
        for s, subfolder_path in enumerate(subfolder):
            print(s)
            if 'nop' in subfolder_path: # or '15rm' in subfolder_path
                continue
            step_data = []
            rew_data = []
            lens = []
            for i in range(num_seeds):
                try:
                    file_path = subfolder_path + '/avg{}.csv'.format(i)
                    data = np.genfromtxt(file_path, delimiter=',')
                    step_data.extend(data[:, 0])
                    rew_data.extend(data[:, 1])
                    lens.append(data[-1, 0])
                except OSError:
                    pass
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
            rew_param.append(avg_rews)
        min_points = min([len(l) for l in rew_param])
        rew_param = np.vstack([rew[:min_points] for rew in rew_param])
        perc_20, perc_80 = np.percentile(rew_param, [20, 80], 0)
        mean = rew_param.mean(axis=0)
        step_plot = np.arange(1, len(mean)+1)*chunk_size
        ax_rew.plot(step_plot, mean, alpha=alpha, c=colors[e], label=keys[e]) #+add_label[e]
        ax_rew.fill_between(step_plot, perc_20, perc_80, color=colors[e], alpha=alpha/3)
        #handles, labels = plt.gca().get_legend_handles_labels()
        #by_label = dict(zip(labels, handles))
        #plt.legend(by_label.values(), by_label.keys())
        #title = subfolder.split('-v2')[0]
        ax_rew.set_title('Humanoid-v2')
    plt.legend(loc='lower right', ncol=1)
    #title += label
    plt.grid()
    fig_rew.savefig(folder_path+title+keys[0])
    plt.close(fig_rew)