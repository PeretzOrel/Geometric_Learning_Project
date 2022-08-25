import numpy as np

from geometric_learning_project import *
from numpy import random, linalg
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from multiprocessing.pool import ThreadPool
import matplotlib.cm as cm
from functions_time import *


def random_ball(num_points, dimension, radius=1):
    random_directions = random.normal(size=(dimension, num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    random_radii = random.random(num_points) ** (1 / dimension)
    return radius * (random_directions * random_radii).T

s = tic()

m = 3  # number of graphs
n = 500  # number of points
d = 3  # dim of a point
k = 6  # num of neighbors

np.random.seed(10)  # set fixed seed

X = np.zeros((m, n, d))

# create set 1 of points:
# r = np.random.uniform(low=0, high=1, size=(n,))
# phi = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
# eta = np.random.uniform(low=0, high=np.pi, size=(n,))
# X[0, :, 0] = r*np.sin(phi)*np.cos(eta)
# X[0, :, 1] = r*np.sin(phi)*np.sin(eta)
# X[0, :, 2] = r*np.cos(phi)

X[0, :, :] = random_ball(n, d, radius=1)

# create set 2 of points:
theta = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
X[1, :, 0], X[1, :, 1] = rotate(X[0, :, 0], X[0, :, 1], theta)
X[1, :, 2] = X[0, :, 2]

# create set 2 of points:
phi = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
X[2, :, 0], X[2, :, 2] = rotate(X[0, :, 0], X[0, :, 2], phi)
X[2, :, 1] = X[0, :, 1]

A = np.zeros((m, n, n))
L = np.zeros((m, n, n))
for i in range(m):
    A[i, :, :] = knn_graph_adjacency_matrix(k, X[i, :, :])
    L[i, :, :] = knn_bi_stochastic_graph_laplacian(A[i, :, :])

# fig. 4
# fig = plt.figure(figsize=plt.figaspect(0.5))
# for i in range(m):
#     ax = fig.add_subplot(1, m, i+1, projection='3d')
#     plt_graph(ax, X[i, :, :], A[i, :, :])
# plt.show(block=False)


# gradient based optimization
t0 = np.array([0, 0])
bounds = ((0, None), (0, None))
cons = ({'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)})
sol = optimize.minimize(func, t0, args=(L,), bounds=bounds, method='SLSQP', jac=func_grad, options={'disp': True, 'ftol': 1e-9}, constraints=cons)
t_opt2 = np.append(sol.x, 1 - np.sum(sol.x))

# t_opt2 = [0.18606561, 0.37422749, 0.4397069 ]
error2 = error_estimate(L, t_opt2)

# fig 5
t1_axis = np.linspace(0, 1, 100)
t2_axis = np.linspace(0, 1, 100)
T1, T2 = np.meshgrid(t1_axis, t2_axis)
Z = np.zeros_like(T1)
Z[Z == 0] = np.NaN

L0 = L[0, :, :] / lambda1(L[0, :, :])
L1 = L[1, :, :] / lambda1(L[1, :, :])
L2 = L[2, :, :] / lambda1(L[2, :, :])

for i in range(T1.shape[0]):
    for j in range(T1.shape[1]):
        if i+j < T1.shape[0]:
            Lt = T1[i, j]*L0 + T2[i, j]*L1 + (1 - T1[i, j] - T2[i, j])*L2
            Z[i, j] = lambda1(Lt)

np.save('Z.npy', Z)
Z = np.load('Z.npy')

fig, ax = plt.subplots(1, 1)
levels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
cpf = ax.contourf(T1, T2, Z, len(levels), cmap=cm.Reds)
line_colors = ['black' for l in cpf.levels]
cp = ax.contour(T1, T2, Z, levels=levels, colors=line_colors)
ax.clabel(cp, fontsize=10, colors=line_colors)
plt.xticks([0, 0.5, 1])
plt.yticks([0, 0.5, 1])
plt.show(block=False)

toc(s)

print()
