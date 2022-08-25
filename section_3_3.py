import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from geometric_learning_project import *
from functions_time import *

s = tic()

n = 250  # number of points
d = 2  # dim of a point
m = 2  # number of graphs
k = 6  # num of neighbors

np.random.seed(1234)  # set fixed seed

X = np.zeros((m, n, d))

# create set 1 of points:
X[0, :, :] = np.random.uniform(low=-0.5, high=0.5, size=(n, d))

# create set 2 of points:
theta = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
X[1, :, 0], X[1, :, 1] = rotate(X[0, :, 0], X[0, :, 1], theta)

A = np.zeros((m, n, n))
L = np.zeros((m, n, n))
for i in range(m):
    A[i, :, :] = knn_graph_adjacency_matrix(k, X[i, :, :])
    L[i, :, :] = knn_bi_stochastic_graph_laplacian(A[i, :, :])

# fig. 1
fig, ax = plt.subplots(1, m)
fig.set_size_inches(10, 5)
for i in range(m):
    plt_graph(ax[i], X[i, :, :], A[i, :, :])
plt.show(block=False)

# brute force optimization
result = optimize.brute(func, ranges=(slice(0, 1, 0.01),), args=(L,), full_output=True, finish=optimize.fmin)
t_opt = np.array([result[0], 1 - result[0]])
error = error_estimate(L, t_opt)

# gradient based optimization
t0 = np.array(0.5)
bounds = ((0, 1),)
result2 = optimize.minimize(func, t0, args=(L,), bounds=bounds, method='trust-constr', jac=func_grad, options={'disp': True})
t_opt2 = [result2.x, 1 - result2.x]
error2 = error_estimate(L, t_opt2)

# fig. 2
plt.figure()
plt.plot(result[2], -result[3], c='black', linewidth=1)
plt.scatter(result[0], -result[1], marker='*')
plt.grid()
plt.xlabel("t_1")
plt.ylabel("lambda _1 ()L_t")
plt.show(block=False)

# fig. 3
r = np.sqrt(X[0, :, 0] ** 2 + X[0, :, 1] ** 2)
idx = r.argsort()

psi1_L_t_opt = psi1(L_t(L, t_opt2))
psi1_L_1 = psi1(L[0, :, :])
psi1_L_2 = psi1(L[1, :, :])

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(15, 5)
ax[0].plot(r[idx], -psi1_L_t_opt[idx], c='black', linewidth=1)
ax[0].grid()
ax[0].set_xlabel('r')
ax[0].set_ylabel('psi1_L_t_opt')

ax[1].plot(r[idx], -psi1_L_1[idx], c='black', linewidth=1)
ax[1].grid()
ax[1].set_xlabel('r')
ax[1].set_ylabel('psi1_L_1')

ax[2].plot(r[idx], -psi1_L_2[idx], c='black', linewidth=1)
ax[2].grid()
ax[2].set_xlabel('r')
ax[2].set_ylabel('psi1_L_2')

plt.show(block=False)

toc(s)

print("")
