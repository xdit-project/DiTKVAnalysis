import numpy as np
import matplotlib.pyplot as plt

models = {
    'flux': {'nlayers': 57, 'nsteps': 28, 'nrows': 19, 'height': 12, 'step': 8, 'exp': 2},
    'pixart-alpha': {'nlayers': 28, 'nsteps': 20, 'nrows': 14, 'height': 8, 'step': 5, 'exp': 2},
    'sd3': {'nlayers': 24, 'nsteps': 28, 'nrows': 12, 'height': 8, 'step': 8, 'exp': 2},
    'cogvideox-5b': {'nlayers': 30, 'nsteps': 50, 'nrows': 15, 'height': 8, 'step': 15, 'exp': 2},
    'latte': {'nlayers': 56, 'nsteps': 50, 'nrows': 19, 'height': 12, 'step': 15, 'exp': 2},
    'opensora': {'nlayers': 56, 'nsteps': 30, 'nrows': 19, 'height': 12, 'step': 10, 'exp': 2},
    'mochi': {'nlayers': 48, 'nsteps': 64, 'nrows': 16, 'height': 12, 'step': 20, 'exp': 2},
    'flux-56': {'nlayers': 57, 'nsteps': 56, 'nrows': 19, 'height': 12, 'step': 20, 'exp': 3},
    'flux-14': {'nlayers': 57, 'nsteps': 14, 'nrows': 19, 'height': 12, 'step': 4, 'exp': 3},
    'flux-sd3-scheduler': {'nlayers': 57, 'nsteps': 28, 'nrows': 19, 'height': 12, 'step': 8, 'exp': 4},
}
batchsize = 10

for model, info in models.items():
    fig, ax = plt.subplots((info['nlayers'] + info['nrows'] - 1) // info['nrows'], info['nrows'], figsize=(48, info['height']))
    L1k = np.zeros((info['nlayers'], info['nsteps'] - 1))
    L1v = np.zeros((info['nlayers'], info['nsteps'] - 1))
    L1a = np.zeros((info['nlayers'], info['nsteps'] - 1))

    for i in range(info['nlayers']):
        for batch in range(batchsize):
            try:
                x = np.load(f'redundancy/{model}/{batch}_l{i}.npy', allow_pickle=True).item()
                if batch == 0:
                    L1k[i] = np.array(x['means']['k'])
                    L1v[i] = np.array(x['means']['v'])
                    L1a[i] = np.array(x['means']['a'])
                else:
                    L1k[i] = L1k[i] * (batch - 1) / batch + np.array(x['means']['k']) / batch
                    L1v[i] = L1v[i] * (batch - 1) / batch + np.array(x['means']['v']) / batch
                    L1a[i] = L1a[i] * (batch - 1) / batch + np.array(x['means']['a']) / batch
            except:
                pass
    maxv = max(np.max(L1k), np.max(L1v), np.max(L1a))
    minv = min(np.min(L1k), np.min(L1v), np.min(L1a))
    for i in range(info['nlayers']):
        row, column = i//info['nrows'], i%info['nrows']
        ax[row, column].plot(range(info['nsteps'] - 2, -1, -1), L1k[i], label='K', linewidth=5, color='mediumpurple')
        ax[row, column].plot(range(info['nsteps'] - 2, -1, -1), L1v[i], label='V', linewidth=5, color='darkorange')
        ax[row, column].plot(range(info['nsteps'] - 2, -1, -1), L1a[i], label='A', linewidth=5, color='cornflowerblue')
        ax[row, column].set_xlim(info['nsteps'] - 1, 0)
        ax[row, column].set_yscale('log')
        if row == (info['nlayers'] + info['nrows'] - 1) // info['nrows'] - 1:
            ax[row, column].set_xticks(range(0, info['nsteps'], info['step']))
            ax[row, column].set_xlabel('Step', fontsize=30)
        else:
            ax[row, column].get_xaxis().set_visible(False)
        ax[row, column].set_ylim(minv, maxv)
        if column == 0:
            ax[row, column].set_ylabel('L1 Distance', fontsize=30)
            ax[row, column].tick_params(axis='both', which='major', labelsize=30)
            ax[row, column].tick_params(axis='both', which='minor', labelsize=25)
        else:
            ax[row, column].get_yaxis().set_visible(False)
            ax[row, column].tick_params(axis='x', which='major', labelsize=30)
            ax[row, column].tick_params(axis='x', which='minor', labelsize=25)
        if i == 0:
            handles, labels = ax[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, facecolor='white', framealpha=0, ncol=3, bbox_to_anchor=(0.5, 1), fontsize=30)
    
    for i in range(info['nlayers'], (info['nlayers'] + info['nrows'] - 1) // info['nrows'] * info['nrows']):
        plt.delaxes(ax[i//info['nrows'], i%info['nrows']])
    
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.8 + info['height'] * 0.01))
    fig.savefig(f'exp-results/{info["exp"]}-{model}.pdf')
    plt.close(fig)
    