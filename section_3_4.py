import numpy as np
from util_functions import *
from numpy import random, linalg
import matplotlib.cm as cm

np.random.seed(10)  # set fixed seed

m = 3  # number of graphs
n = 500  # number of points
d = 3  # dim of a point
k = 6  # num of neighbors

# create set 1 of points:
X = np.zeros((m, n, d))
r = np.random.uniform(low=0, high=1, size=(n,))**(1/3)
phi = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
cos_theta = np.random.uniform(low=-1, high=1, size=(n,))
X[0, :, 0] = r*np.sqrt(1-cos_theta**2)*np.cos(phi)
X[0, :, 1] = r*np.sqrt(1-cos_theta**2)*np.sin(phi)
X[0, :, 2] = r*cos_theta

# create set 2 of points:
theta = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
X[1, :, 0], X[1, :, 1] = rotate(X[0, :, 0], X[0, :, 1], theta)
X[1, :, 2] = X[0, :, 2]

# create set 3 of points:
phi = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
X[2, :, 0], X[2, :, 2] = rotate(X[0, :, 0], X[0, :, 2], phi)
X[2, :, 1] = X[0, :, 1]

A = knn_graph_adjacency_matrix(k, X)
L = knn_bi_stochastic_graph_laplacian(A)

# fig. 4
plt.rcParams['text.usetex'] = True
fig = plt.figure()
fig.set_size_inches(15, 5)

ax = fig.add_subplot(1, 3, 1, projection='3d')
plt_graph(ax, X[0, :, :], A[0, :, :])
ax.set_title(r"$6-NN\ Graph,\ Unit\ Ball$", fontsize=8)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")

ax = fig.add_subplot(1, 3, 2, projection='3d')
plt_graph(ax, X[1, :, :], A[1, :, :])
ax.set_title(r"$6-NN\ Graph,\ z-axis\ Rotation$", fontsize=8)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")

ax = fig.add_subplot(1, 3, 3, projection='3d')
plt_graph(ax, X[2, :, :], A[2, :, :])
ax.set_title(r"$6-NN\ Graph,\ y-axis\ Rotation$", fontsize=8)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$z$")

plt.show(block=False)
# fig.savefig('fig4.svg', format='svg')

# gradient based optimization
t0 = np.array([0, 0])
bounds = ((0, None), (0, None))
cons = ({'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)})
sol = optimize.minimize(func, t0, args=(L,), bounds=bounds, method='SLSQP', jac=func_grad, options={'disp': True, 'ftol': 1e-9}, constraints=cons)
t_opt = np.append(sol.x, 1 - np.sum(sol.x))
error2 = error_estimate(L, t_opt)

# fig 5
T1, T2 = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
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

plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(1, 1)
levels = np.array([0, 1, 2, 3, 4, 5, 5.5, 6, 7, 8, 9, 10])
cpf = ax.contourf(T1, T2, Z, len(levels), cmap=cm.Reds)
line_colors = ['black' for l in cpf.levels]
cp = ax.contour(T1, T2, Z, levels=levels, colors=line_colors)
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
# fig.savefig('fig5.svg', format='svg')

# fig 6
r = np.sqrt(X[0, :, 0] ** 2 + X[0, :, 1] ** 2 + X[0, :, 2] ** 2)
idx = r.argsort()

psi1_L_t_opt = psi1(L_t(L, t_opt))
psi1_L_1 = psi1(L[0, :, :])
psi1_L_2 = psi1(L[1, :, :])
psi1_L_3 = psi1(L[2, :, :])

plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(1, 4)
fig.set_size_inches(25, 5)

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

ax[3].plot(r[idx], psi1_L_3[idx], c='black', linewidth=0.7)
ax[3].set_xlabel(r"$r$")
ax[3].set_ylabel(r"$\psi_1(L_3)$")
ax[3].grid()

plt.show(block=False)
# fig.savefig('fig6.svg', format='svg')
