import numpy as np
from scipy.spatial.distance import pdist, squareform
from sinkhorn_knopp import sinkhorn_knopp as skp
import matplotlib.pyplot as plt
from scipy import optimize


def rotate_points_2d(points_2d, theta):
    points_2d_rotated = np.zeros_like(points_2d)
    r = np.sqrt(np.sum(points_2d ** 2, axis=1))
    phi = np.arctan2(points_2d[:, 1], points_2d[:, 0])
    points_2d_rotated[:, 0] = r * np.cos(phi + theta)
    points_2d_rotated[:, 1] = r * np.sin(phi + theta)
    return points_2d_rotated


def my_pdist(k, X):
    dist = squareform(pdist(X, metric='euclidean'))
    sorted_idx = dist.argsort(axis=0)
    sorted_dist = dist[sorted_idx, np.arange(sorted_idx.shape[1])]
    knn_idx = sorted_idx[1:k + 1, :]
    knn_dist = sorted_dist[1:k + 1, :]
    return knn_idx, knn_dist


def knn_graph_adjacency_matrix(k, X):
    knn_idx, knn_dist = my_pdist(k, X)
    # construct adjacency matrix A
    A = np.identity(n)
    for i in range(n):
        A[i, knn_idx[:, i]] = 1
        A[knn_idx[:, i], i] = 1
    return A


def knn_bi_stochastic_graph_laplacian(k, A):
    sk = skp.SinkhornKnopp(max_iter=2000, epsilon=1e-5)
    B = sk.fit(A)
    return np.identity(n) - B


def plt_graph(ax, X, A):
    edges = np.transpose(np.array(A.nonzero()))
    edges = np.delete(edges, edges[:, 0] == edges[:, 1], axis=0)

    ax.scatter(X[:, 0], X[:, 1], s=20, c="red", edgecolor="red")
    ax.grid()
    for i in range(0, edges.shape[0], 1):
        ax.plot([X[edges[i, 0], 0], X[edges[i, 1], 0]], [X[edges[i, 0], 1], X[edges[i, 1], 1]], c='black', linewidth=1)


def sorted_eig(L):
    eig_val, eig_vec = np.linalg.eig(L)
    idx = eig_val.argsort()
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]
    return eig_val, eig_vec


def lambda1(L):
    eig_val, eig_vec = sorted_eig(L)
    return eig_val[1].real


def psi1(L):
    eig_val, eig_vec = sorted_eig(L)
    return eig_vec[:, 1].real


def S_L(L, x):
    return (np.transpose(x) @ L @ x) / lambda1(L)


def L_t(L, t):
    m, n, n = L.shape
    Lt = np.zeros((n, n))
    for k in range(m):
        Lt += (t[k] * L[k, :, :]) / lambda1(L[k, :, :])
    return Lt


def func(t0, args):
    L = args
    t = (t0, 1 - t0)
    return -lambda1(L_t(L, t))


def func_grad(t0, args):
    L = args
    t = (t0, 1 - t0)
    m, n, n = L.shape
    grad = np.zeros_like(t0)
    for k in range(m - 1):
        grad[k] = np.transpose(psi1(L_t(L, t))) @ (L[k, :, :] / lambda1(L[k, :, :]) - L[m - 1, :, :] / lambda1(L[m - 1, :, :])) @ psi1(L_t(L, t))
    return -grad


def error_estimate(L, t_opt):
    m, n, n = L.shape
    max_score = -np.inf
    for k in range(m):
        score = S_L(L[k, :, :], psi1(L_t(L, t_opt)))
        if score > max_score:
            max_score = score
    return np.abs(max_score - lambda1(L_t(L, t_opt)))


n = 250  # number of points
d = 2  # dim of a point
m = 2  # number of graphs
k = 6  # num of neighbors

np.random.seed(265)  # set fixed seed

X = np.zeros((m, n, d))
# create set 1 of points:
X[0, :, :] = np.random.uniform(low=-0.5, high=0.5, size=(n, d))

# create set 2 of points:
theta = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
X[1, :, :] = rotate_points_2d(X[0, :, :], theta)

A = np.zeros((m, n, n))
L = np.zeros((m, n, n))
for i in range(m):
    A[i, :, :] = knn_graph_adjacency_matrix(k, X[i, :, :])
    L[i, :, :] = knn_bi_stochastic_graph_laplacian(k, A[i, :, :])

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
result2 = optimize.minimize(func, t0, args=(L,), bounds=bounds, method='SLSQP', jac=func_grad, options={'ftol': 1e-10, 'disp': True})
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
ax[0].plot(r[idx], psi1_L_t_opt[idx], c='black', linewidth=1)
ax[0].grid()
ax[0].set_xlabel('r')
ax[0].set_ylabel('psi1_L_t_opt')

ax[1].plot(r[idx], psi1_L_1[idx], c='black', linewidth=1)
ax[1].grid()
ax[1].set_xlabel('r')
ax[1].set_ylabel('psi1_L_1')

ax[2].plot(r[idx], psi1_L_2[idx], c='black', linewidth=1)
ax[2].grid()
ax[2].set_xlabel('r')
ax[2].set_ylabel('psi1_L_2')

plt.show(block=False)

print("")
