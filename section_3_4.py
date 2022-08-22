from geometric_learning_project import *
from numpy import random, linalg
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint


def random_ball(num_points, dimension, radius=1):
    random_directions = random.normal(size=(dimension,num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    random_radii = random.random(num_points) ** (1/dimension)
    return radius * (random_directions * random_radii).T


def z_axis_rotation(cart_3d_points, theta0):
    cart_3d_points_rotated = np.zeros_like(cart_3d_points)
    cart_3d_points_rotated[:, 0] = cart_3d_points[:, 0] * np.cos(theta0) - cart_3d_points[:, 1]*np.sin(theta0)
    cart_3d_points_rotated[:, 1] = cart_3d_points[:, 0] * np.sin(theta0) + cart_3d_points[:, 1]*np.cos(theta0)
    cart_3d_points_rotated[:, 2] = cart_3d_points[:, 2]
    return cart_3d_points_rotated


def y_axis_rotation(cart_3d_points, phi0):
    cart_3d_points_rotated = np.zeros_like(cart_3d_points)
    cart_3d_points_rotated[:, 0] = cart_3d_points[:, 0] * np.cos(phi0) + cart_3d_points[:, 2]*np.sin(phi0)
    cart_3d_points_rotated[:, 1] = cart_3d_points[:, 1]
    cart_3d_points_rotated[:, 2] = - cart_3d_points[:, 0] * np.sin(phi0) + cart_3d_points[:, 2] * np.cos(phi0)
    return cart_3d_points_rotated


n = 500  # number of points
d = 3  # dim of a point
m = 3  # number of graphs
k = 6  # num of neighbors

np.random.seed(265)  # set fixed seed

X = np.zeros((m, n, d))

# create set 1 of points:
X[0, :, :] = random_ball(n, d, radius=1)

# create set 2 of points:
dtheta = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
X[1, :, :] = z_axis_rotation(X[0, :, :], dtheta)

# create set 2 of points:
dphi = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
X[2, :, :] = y_axis_rotation(X[0, :, :], dphi)


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
cons = [{'type': 'ineq', 'fun': lambda x:  1 - x[0] - x[1]}]
# result2 = optimize.minimize(func, t0, args=(L,), bounds=bounds, method='SLSQP', jac=func_grad, options={'disp': True}, constraints=cons)

# t_opt2 = np.append(result2.x, 1 - np.sum(result2.x))
t_opt2 = [0.15531945, 0.48021447, 0.36446609]
error2 = error_estimate(L, t_opt2)

# fig 5
t1 = np.linspace(0, 1, 100)
t2 = np.linspace(0, 1, 100)
T1, T2 = np.meshgrid(t1, t2)

Z = np.zeros_like(T1)
for i in range(T1.shape[0]):
    for j in range(T1.shape[1]):
        Z[i, j] = lambda1(L_t(L, (T1[i, j], T2[i, j], 1 - T1[i, j] - T2[i, j])))
np.save('Z.npy', Z)
# fig, ax = plt.subplots(1, 1)
# ax.contour(T1, T2, Z)
# plt.show(block=False)

print()
