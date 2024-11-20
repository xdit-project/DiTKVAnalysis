import numpy as np
import matplotlib.pyplot as plt

models = {
    'flux': {'nlayers': 57, 'nsteps': 28},
    'pixart-alpha': {'nlayers': 28, 'nsteps': 20},
    'sd3': {'nlayers': 24, 'nsteps': 28},
    'cogvideox-5b': {'nlayers': 30, 'nsteps': 50},
    'latte': {'nlayers': 56, 'nsteps': 50},
    'opensora': {'nlayers': 56, 'nsteps': 30},
}
batchsize = 10

for model, info in models.items():
    fig, ax = plt.subplots(8, 8, figsize=(48, 32))
    fig2, ax2 = plt.subplots(8, 8, figsize=(48, 32))
    L1k = np.zeros((batchsize, info['nlayers'], info['nsteps'] - 1))
    L1v = np.zeros((batchsize, info['nlayers'], info['nsteps'] - 1))
    L1a = np.zeros((batchsize, info['nlayers'], info['nsteps'] - 1))

    for i in range(info['nlayers']):
        for batch in range(batchsize):
            info = np.load(f'redundancy/{model}/{batch}_l{i}.npy', allow_pickle=True).item()
            L1k[batch, i] = np.array(info['means']['k'])
            L1v[batch, i] = np.array(info['means']['v'])
            L1a[batch, i] = np.array(info['means']['a'])
    
    stdk = np.mean(np.sqrt(np.var(L1k, axis=0)) / np.mean(L1k, axis=0))
    stdv = np.mean(np.sqrt(np.var(L1v, axis=0)) / np.mean(L1v, axis=0))
    stda = np.mean(np.sqrt(np.var(L1a, axis=0)) / np.mean(L1a, axis=0))
    print(model, "{:.3f}".format(stdk), "{:.3f}".format(stdv), "{:.3f}".format(stda))
