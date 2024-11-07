import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(8, 8, figsize=(48, 32))

for i in range(57):
    info = np.load(f'redundancy/flux/3_l{i}.npy', allow_pickle=True).item()

    row, column = i//8, i%8
    ax[row, column].errorbar(range(1, 28), info['means']['k'], yerr=np.sqrt(info['vars']['k']), 
                    label='Key Diff', color='blue', capsize=5)

    ax[row, column].errorbar(range(1, 28), info['means']['v'], yerr=np.sqrt(info['vars']['v']), 
                    label='Value Diff', color='red', capsize=5)
    ax[row, column].set_xticks(range(0, 28, 4))
    ax[row, column].set_xlabel('Timestep')
    ax[row, column].set_ylabel('Mean of Absolute Differences')
    ax[row, column].legend()
    ax[row, column].set_title(f'Attention {i} KV Diff')

fig.tight_layout()
fig.savefig('3.pdf')