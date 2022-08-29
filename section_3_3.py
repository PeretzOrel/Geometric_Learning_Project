import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from util_functions import *

np.random.seed(10)  # set fixed seed

n = 250  # number of points
d = 2  # dim of a point
m = 2  # number of graphs
k = 6  # num of neighbors

# create set 1 of points:
X = np.zeros((m, n, d))
X[0, :, :] = np.random.uniform(low=-0.5, high=0.5, size=(n, d))

# create set 2 of points:
theta = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
X[1, :, 0], X[1, :, 1] = rotate(X[0, :, 0], X[0, :, 1], theta)

A = knn_graph_adjacency_matrix(k, X)
L = knn_bi_stochastic_graph_laplacian(A)

# fig. 1
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(1, m)
fig.set_size_inches(10, 5)

plt_graph(ax[0], X[0, :, :], A[0, :, :])
ax[0].set_title(r"$6-NN\ Graph$", fontsize=8)
ax[0].set_xlabel(r"$x$")
ax[0].set_ylabel(r"$y$")
ax[0].set_aspect(1)

plt_graph(ax[1], X[1, :, :], A[1, :, :])
ax[1].set_title(r"$6-NN\ Graph,\ after\ rotation$", fontsize=8)
ax[1].set_xlabel(r"$x$")
ax[1].set_ylabel(r"$y$")
ax[1].set_aspect(1)
plt.show(block=False)
# fig.savefig('fig1.svg', format='svg')

# Optimization
result = optimize.brute(func, ranges=(slice(0, 1, 0.01),), args=(L,), full_output=True, finish=optimize.fmin)
t_opt = np.array([result[0], 1 - result[0]])
error = error_estimate(L, t_opt)

t0 = np.array(0.5)
bounds = ((0, 1),)
result2 = optimize.minimize(func, t0, args=(L,), bounds=bounds, method='trust-constr', jac=func_grad, options={'disp': True})
t_opt2 = [result2.x, 1 - result2.x]
error2 = error_estimate(L, t_opt2)

# fig. 2
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(1, 1)
ax.plot(result[2], -result[3], c='black', linewidth=0.7)
ax.scatter(result[0], -result[1], marker='*', c='k')
ax.set_title(r"$\lambda_1(L_t) \ as\ Function\ of\ t_1 $", fontsize=8)
ax.set_xlabel(r"$t_1$")
ax.set_ylabel(r"$\lambda_1(L_t)$")
ax.grid()
plt.show(block=False)
# fig.savefig('fig2.svg', format='svg')

# fig. 3
r = np.sqrt(X[0, :, 0] ** 2 + X[0, :, 1] ** 2)
idx = r.argsort()

psi1_L_t_opt = psi1(L_t(L, t_opt2))
psi1_L_1 = psi1(L[0, :, :])
psi1_L_2 = psi1(L[1, :, :])

plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(20, 5)

ax[0].plot(r[idx], psi1_L_t_opt[idx], c='black', linewidth=0.7)
ax[0].set_xlabel(r"$r$")
ax[0].set_ylabel(r"$\psi_1(L_{t^*})$")
ax[0].grid()

ax[1].plot(r[idx], psi1_L_1[idx], c='black', linewidth=0.7)
ax[1].set_xlabel(r"$r$")
ax[1].set_ylabel(r"$\psi_1(L_1)$")
ax[1].grid()

ax[2].plot(r[idx], psi1_L_2[idx], c='black', linewidth=0.7)
ax[2].set_xlabel(r"$r$")
ax[2].set_ylabel(r"$\psi_1(L_2)$")
ax[2].grid()

plt.show(block=False)
# fig.savefig('fig3.svg', format='svg')
