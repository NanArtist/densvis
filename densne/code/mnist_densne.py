import os
import sys
sys.path.append("../densne")
sys.path.append("..")

import numpy as np
import densne

from sklearn.datasets import fetch_openml
from sklearn.utils import resample

import matplotlib.pyplot as plt
import matplotlib.cm as cm

digits = fetch_openml(name='mnist_784')

subsample, subsample_labels = resample(digits.data, digits.target, n_samples=7000,
                                       stratify=digits.target, random_state=1)

# subsample, subsample_labels = digits.data, digits.target


# denSNE run
print('Running denSNE')
emb, ro, re = densne.run_densne(subsample, use_pca=False)
dens = np.stack((ro, re)).transpose()


# tSNE run
print('Running tSNE')
t_emb, t_ro, t_re = densne.run_densne(subsample, perplexity=50, verbose=True, initial_dims=subsample.shape[1],
                               dens_frac=0, use_pca=False, max_iter=1000, dens_lambda=0,
                               final_dens=True)
t_dens = np.stack((t_ro,t_re)).transpose()


# Saving
if not os.path.exists('mnist'): os.mkdir('mnist')
print('-- Saving denSNE --')
print('emb: densne_mnist_emb.txt')
print('Densities: densne_mnist_dens.txt')
np.savetxt('../mnist/densne_mnist_emb.txt', emb)
np.savetxt('../mnist/densne_mnist_dens.txt', dens)


print('-- Saving tSNE --')
print('emb: tsne_mnist_emb.txt')
print('Densities: tsne_mnist_dens.txt')
np.savetxt('../mnist/tsne_mnist_emb.txt', emb)
np.savetxt('../mnist/tsne_mnist_dens.txt', dens)


# Plotting
print('Plotting')
K=10
N=subsample.shape[0]
color_int = np.array(subsample_labels, dtype=int)
cmap = cm.viridis(np.linspace(0,1,K))
colors=[cmap[i] for i in color_int]


fig, ax = plt.subplots(2,2,figsize=[10,10])

x_0 = np.mean(embed[np.where(color_int == 0), :])
y_0 = np.mean(embed[:, np.where(color_int == 1)])

x_0 = np.mean(embed[np.where(color_int == 0), :])
y_0 = np.mean(embed[:, np.where(color_int == 1)])


ax[0,0].scatter(emb[:,0], emb[:,1], c=colors, s=2, alpha=.4)
ax[0,1].scatter(dens[:,0], dens[:,1], c=colors, s=2, alpha=.4)

ax[1,0].scatter(t_emb[:,0], t_emb[:,1], c=colors, s=2, alpha=.4)
ax[1,1].scatter(t_dens[:,0], t_dens[:,1], c=colors, s=2, alpha=.4)


ax[0,0].set_ylabel('denSNE_y')
ax[1,0].set_ylabel('tSNE_y')

ax[0,0].set_xlabel('denSNE_x')
ax[1,0].set_xlabel('tSNE_x')


ax[0,0].set_title('emb (den-SNE)')
ax[1,0].set_title('emb (t-SNE)')
ax[0,1].set_title('Density Preservation (den-SNE)')
ax[1,1].set_title('Density Preservation (t-SNE)')

ax[1,1].set_xlabel('Original Local Radius (log)')
ax[0,1].set_xlabel('Original Local Radius (log)')

ax[0,1].set_ylabel('emb Local Radius (log)')
ax[1,1].set_ylabel('emb Local Radius (log)')

ax[0,0].set_xticklabels('')
ax[0,0].set_xticks([])
ax[0,0].set_yticklabels('')
ax[0,0].set_yticks([])

ax[1,0].set_xticklabels('')
ax[1,0].set_xticks([])
ax[1,0].set_yticklabels('')
ax[1,0].set_yticks([])

fig.savefig('../mnist/densne_mnist_fig.png', bbox_inches='tight')

plt.show()
