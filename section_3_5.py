import numpy as np

from geometric_learning_project import *
from numpy import random, linalg
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from multiprocessing.pool import ThreadPool
import matplotlib.cm as cm
from functions_time import tic, toc

s = tic()


m = 3  # number of graphs
n = 250  # number of points
d = 2  # dim of a point
k = 6  # num of neighbors

np.random.seed(10)  # set fixed seed

X = np.zeros((m, n, d))

# create set 1 of points:
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

A = np.zeros((m, n, n))
L = np.zeros((m, n, n))
for i in range(m):
    A[i, :, :] = knn_graph_adjacency_matrix(k, X[i, :, :])
    L[i, :, :] = knn_bi_stochastic_graph_laplacian(A[i, :, :])

# fig. 7
# fig, ax = plt.subplots(1, m)
# fig.set_size_inches(10, 5)
# for i in range(m):
#     plt_graph(ax[i], X[i, :, :], A[i, :, :])
# plt.show(block=False)

# fig 8
T1, T2 = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
Z2 = np.zeros_like(T1)
Z2[Z2 == 0] = np.NaN

L0 = L[0, :, :] / lambda1(L[0, :, :])
L1 = L[1, :, :] / lambda1(L[1, :, :])
L2 = L[2, :, :] / lambda1(L[2, :, :])

# for i in range(T1.shape[0]):
#     for j in range(T1.shape[1]):
#         if i+j < T1.shape[0]:
#             Lt = T1[i, j]*L0 + T2[i, j]*L1 + (1 - T1[i, j] - T2[i, j])*L2
#             Z2[i, j] = lambda1(Lt)
#
# np.save('Z2.npy', Z2)
Z2 = np.load('Z2.npy')

# fig, ax = plt.subplots(1, 1)
# levels = np.array([5, 10, 15, 20, 25, 30, 35, 40])
# cpf = ax.contourf(T1, T2, Z2, len(levels), cmap=cm.Reds)
# line_colors = ['black' for l in cpf.levels]
# cp = ax.contour(T1, T2, Z2, levels=levels, colors=line_colors)
# ax.clabel(cp, fontsize=10, colors=line_colors)
# plt.xticks([0, 0.5, 1])
# plt.yticks([0, 0.5, 1])
# plt.show(block=False)

# gradient based optimization
t0 = np.array([0, 0])
bounds = ((0, None), (0, None))
cons = ({'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)})
sol = optimize.minimize(func, t0, args=(L,), bounds=bounds, method='SLSQP', jac=func_grad, options={'disp': True, 'ftol': 1e-15}, constraints=cons)
t_opt = np.append(sol.x, 1 - np.sum(sol.x))

# t_opt = [8.33119543e-13, 4.76667284e-01, 5.23332716e-01]
error2 = error_estimate(L, t_opt)

toc(s)


print()
