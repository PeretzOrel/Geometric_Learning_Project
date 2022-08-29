import numpy as np
from util_functions import *
from numpy import random, linalg
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from multiprocessing.pool import ThreadPool
import matplotlib.cm as cm

np.random.seed(10)  # set fixed seed

m = 3  # number of graphs
n = 250  # number of points
d = 2  # dim of a point
k = 6  # num of neighbors

# create set 1 of points:
X = np.zeros((m, n, d))
r = np.sqrt(np.random.uniform(low=0, high=1, size=(n,)))
theta = np.random.uniform(low=0, high=2*np.pi, size=(n,))
X[0, :, 0] = r*np.cos(theta)
X[0, :, 1] = r*np.sin(theta)

# create set 2 of points:
X[1, :, 0] = X[0, :, 0]
X[1, :, 1] = X[0, :, 1] * (1-np.cos(np.pi*X[0, :, 0]))

# create set 3 of points:
X[2, :, 0] = X[0, :, 0] * (1-np.cos(np.pi*X[0, :, 1]))
X[2, :, 1] = X[0, :, 1]

A = knn_graph_adjacency_matrix(k, X)
L = knn_bi_stochastic_graph_laplacian(A)

# fig. 7
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(1, m)
fig.set_size_inches(15, 5)

plt_graph(ax[0], X[0, :, :], A[0, :, :])
ax[0].set_title(r"$6-NN\ Graph,\ Unit\ Disc$", fontsize=8)
ax[0].set_xlabel(r"$x$")
ax[0].set_ylabel(r"$y$")
ax[0].set_aspect(1)

plt_graph(ax[1], X[1, :, :], A[1, :, :])
ax[1].set_title(r"$6-NN\ Graph,\ Horizontal\ Barbell\ Squeeze$", fontsize=8)
ax[1].set_xlabel(r"$x$")
ax[1].set_ylabel(r"$y$")
ax[1].set_aspect(1)

plt_graph(ax[2], X[2, :, :], A[2, :, :])
ax[2].set_title(r"$6-NN\ Graph,\ Vertical\ Barbell\ Squeeze$", fontsize=8)
ax[2].set_xlabel(r"$x$")
ax[2].set_ylabel(r"$y$")
ax[2].set_aspect(1)

plt.show(block=False)
# fig.savefig('fig7.svg', format='svg')

# gradient based optimization
t0 = np.array([0, 0])
bounds = ((0, None), (0, None))
cons = ({'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)})
sol = optimize.minimize(func, t0, args=(L,), bounds=bounds, method='SLSQP', jac=func_grad, options={'disp': True, 'ftol': 1e-15}, constraints=cons)
t_opt = np.append(sol.x, 1 - np.sum(sol.x))
error2 = error_estimate(L, t_opt)

# fig 8
T1, T2 = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
Z2 = np.zeros_like(T1)
Z2[Z2 == 0] = np.NaN

L0 = L[0, :, :] / lambda1(L[0, :, :])
L1 = L[1, :, :] / lambda1(L[1, :, :])
L2 = L[2, :, :] / lambda1(L[2, :, :])

for i in range(T1.shape[0]):
    for j in range(T1.shape[1]):
        if i+j < T1.shape[0]:
            Lt = T1[i, j]*L0 + T2[i, j]*L1 + (1 - T1[i, j] - T2[i, j])*L2
            Z2[i, j] = lambda1(Lt)

plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(1, 1)
levels = np.array([5, 10, 15, 20, 25, 30, 35, 40])
cpf = ax.contourf(T1, T2, Z2, len(levels), cmap=cm.Reds)
line_colors = ['black' for l in cpf.levels]
cp = ax.contour(T1, T2, Z2, levels=levels, colors=line_colors)
ax.clabel(cp, fontsize=10, colors=line_colors)
plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5, 1])
ax.set_title(r"$Contour\ plot\ of\ \lambda_1(L_t) $", fontsize=8)
ax.set_xlabel(r"$t_1$")
ax.set_ylabel(r"$t_2$")
ax.scatter(t_opt[0], t_opt[1], marker='*', c='k')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show(block=False)
# fig.savefig('fig8.svg', format='svg')

# fig 9
NE = (X[0, :, 0] >= 0) & (X[0, :, 1] >= 0)
NW = (X[0, :, 0] < 0) & (X[0, :, 1] > 0)
SW = (X[0, :, 0] < 0) & (X[0, :, 1] < 0)
SE = (X[0, :, 0] >= 0) & (X[0, :, 1] < 0)

Lt = L_t(L, t_opt)
eig_val, eig_vec = sp.linalg.eigh(Lt, subset_by_index=[1, 3])

plt.rcParams['text.usetex'] = True
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(eig_vec[NE, 0], eig_vec[NE, 1], eig_vec[NE, 2], s=10, c='black', label="NE")
ax.scatter(eig_vec[NW, 0], eig_vec[NW, 1], eig_vec[NW, 2], s=10, c='red', label="NW")
ax.scatter(eig_vec[SW, 0], eig_vec[SW, 1], eig_vec[SW, 2], s=10, c='blue', label="SW")
ax.scatter(eig_vec[SE, 0], eig_vec[SE, 1], eig_vec[SE, 2], s=10, c='orange', label="SE")
ax.set_xlabel(r"$\psi_1$")
ax.set_ylabel(r"$\psi_2$")
ax.set_zlabel(r"$\psi_3$")
plt.legend(loc="upper left")
plt.show(block=False)
# fig.savefig('fig9.svg', format='svg')
