import numpy as np
import scipy as sp
from scipy.spatial.distance import pdist, squareform
from scipy import optimize
import matplotlib.pyplot as plt

def rotate(x, y, ang):
    r = np.sqrt(x ** 2 + y ** 2)
    # theta = np.arctan2(x, y)
    theta = np.arctan(y / x)
    x_out = r * np.cos(theta + ang)
    y_out = r * np.sin(theta + ang)
    return x_out, y_out


def my_pdist(k, X):
    dist = squareform(pdist(X, metric='euclidean'))
    sorted_idx = dist.argsort(axis=0)
    sorted_dist = dist[sorted_idx, np.arange(sorted_idx.shape[1])]
    knn_idx = sorted_idx[1:k + 1, :]
    knn_dist = sorted_dist[1:k + 1, :]
    return knn_idx, knn_dist


def knn_graph_adjacency_matrix(k, X):
    (m, n, d) = X.shape
    A = np.zeros((m, n, n))
    for graph_idx in range(m):
        knn_idx, knn_dist = my_pdist(k, X[graph_idx, :, :])
        A_tmp = np.identity(n)
        for i in range(n):
            A_tmp[i, knn_idx[:, i]] = 1
            A_tmp[knn_idx[:, i], i] = 1
        A[graph_idx, :, :] = A_tmp
    return A


def knn_bi_stochastic_graph_laplacian(A):
    (m, n, n) = A.shape
    L = np.zeros((m, n, n))
    for graph_idx in range(m):
        epsilon = 10 ** -6
        Q_i = np.identity(n)
        Q_i_plus_one = np.diag(A[graph_idx, :, :] @ np.linalg.pinv(Q_i) @ np.ones(n))
        for i in range(1000):
            Q_i = np.diag(A[graph_idx, :, :] @ np.linalg.pinv(Q_i_plus_one) @ np.ones(n))
            Q_i_plus_one = np.diag(A[graph_idx, :, :] @ np.linalg.pinv(Q_i) @ np.ones(n))
            D = Q_i_plus_one @ Q_i

            B = sp.linalg.fractional_matrix_power(D, -0.5) @ A[graph_idx, :, :] @ sp.linalg.fractional_matrix_power(D, -0.5)
            error = (np.abs(B @ np.ones(n) - np.ones(n)) < epsilon)
            L_tmp = np.identity(n) - B
            if np.all(error):
                break
        L[graph_idx, :, :] = L_tmp
    return L


def plt_graph(ax, X, A):
    n, d = X.shape
    if d == 2:
        edges = np.transpose(np.array(A.nonzero()))
        edges = np.delete(edges, edges[:, 0] == edges[:, 1], axis=0)

        ax.scatter(X[:, 0], X[:, 1], s=10, c="black", edgecolor="black")
        ax.grid()
        for i in range(0, edges.shape[0], 1):
            ax.plot([X[edges[i, 0], 0], X[edges[i, 1], 0]], [X[edges[i, 0], 1], X[edges[i, 1], 1]], c='black', linewidth=0.7)
    elif d == 3:
        edges = np.transpose(np.array(A.nonzero()))
        edges = np.delete(edges, edges[:, 0] == edges[:, 1], axis=0)

        ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], s=10, c="black", edgecolor="black")
        ax.grid()
        for i in range(0, edges.shape[0], 1):
            ax.plot3D([X[edges[i, 0], 0], X[edges[i, 1], 0]], [X[edges[i, 0], 1], X[edges[i, 1], 1]], [X[edges[i, 0], 2], X[edges[i, 1], 2]],
                      c='black', linewidth=0.7)

        print()


def sorted_eig(L):
    eig_val, eig_vec = np.linalg.eig(L)
    idx = eig_val.argsort()
    eig_val = eig_val[idx]
    eig_vec = eig_vec[:, idx]
    return eig_val, eig_vec


def lambda1(L):
    eig_val = sp.linalg.eigh(L, eigvals_only=True, subset_by_index=[1, 1])
    return eig_val[0]


def psi1(L):
    eig_val, eig_vec = sp.linalg.eigh(L, subset_by_index=[1, 1])
    return eig_vec.flatten()


def S_L(L, x):
    return (np.transpose(x) @ L @ x) / lambda1(L)


def L_t(L, t):
    m, n, n = L.shape
    Lt = np.zeros((n, n))
    for k in range(m):
        Lt += (t[k] * L[k, :, :]) / lambda1(L[k, :, :])
    return Lt


def func(t0, L):
    t = np.append(t0, 1 - np.sum(t0))
    return -lambda1(L_t(L, t))


def func_grad(t0, L):
    t = np.append(t0, 1 - np.sum(t0))
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
