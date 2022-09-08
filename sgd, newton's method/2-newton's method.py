# Newton method with backtracking line search

# x = (x1, x2)
# f(x1, x2) = e^(x1+3x2−0.1) + e^(x1−3x2−0.1) + e^(−x1−0.1)

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

def function(x):
    y = math.exp(x[0]+(3*x[1])-0.1) + math.exp(x[0]-(3*x[1])-0.1) + math.exp(-x[0]-0.1)
    return y


def get_gradient(x):
    dydx1 = math.exp(x[0]+(3*x[1])-0.1) + math.exp(x[0]-(3*x[1])-0.1) - math.exp(-x[0]-0.1)
    dydx2 = (3*math.exp(x[0]+(3*x[1])-0.1)) - (3*math.exp(x[0]-(3*x[1])-0.1))
    return np.array([dydx1, dydx2])


def get_hessian(x):
  N = x.shape[0]
  hessian = np.zeros((N,N)) 
  gd_0 = get_gradient(x)
  eps = np.linalg.norm(gd_0) * np.finfo(np.float32).eps 
  for i in range(N):
    x0 = 1.*x[i]
    x[i] = x0 + eps
    gd_1 =  get_gradient(x)
    hessian[:,i] = ((gd_1 - gd_0)/eps).reshape(x.shape[0])
    x[i] = x0
  return hessian

    
def get_stepsize(type, x_current, dir, init_alpha):
    tau = 0.7
    beta = 0.1
    alpha = init_alpha
    current_grad = get_gradient(x_current)
    current_f = function(x_current)
    x_next = x_current + alpha * (dir)
    while True:
        if np.isnan(function(x_next)):
            alpha *= tau
        else:
            if function(x_next) >= current_f + beta * current_grad.dot(x_next - x_current):
                alpha *= tau
            else:
                break
        x_next = x_current + alpha * (dir)
    return alpha


def solve(initial_x):
    eps = 1e-6
    init_alpha = 1
    max_iter = 1000

    x_vals = []
    f_vals = []
    x_current = initial_x.copy()

    iteration = 0
    current_grad = None
    typ = "backtracking"

    x_vals.append(initial_x)
    f_vals.append(function(initial_x))

    while True:
        if iteration >= max_iter:
            print("Maximum iteration exceeded.")
            break
        current_grad = get_gradient(x_current)
        if np.linalg.norm(current_grad) < eps:
            print("Required tolerance is achieved.")
            break
        alpha = get_stepsize(typ, x_current, -current_grad, init_alpha)
        hessian = get_hessian(x_current)
        hessian_inv = np.linalg.inv(hessian)
    
        x_next = x_current - alpha * np.matmul(hessian_inv, current_grad)
        x_vals.append(x_next)
        f_vals.append(function(x_next))
        x_current = x_next
        iteration += 1

    print("Method used: Newton's Method with {} line search".format(typ))
    print("Value of x1, x2: {}, {}".format(x_current[0], x_current[1]))
    print("Number of iterations: {}".format(iteration))
    print("Function value: {}".format(function(x_current)))
    return x_vals, f_vals


if __name__== "__main__":
    x_0 = np.array([0.5, 0.5])
    x, func = solve(x_0)

    tmp1 = [val[0] for val in x]
    # print(tmp1)
    tmp2 = [val[1] for val in x]
    # print(tmp2)
    
    x1 = np.arange(np.min(tmp1)-0.05, np.max(tmp1)+0.05, 0.01)
    x2 = np.arange(np.min(tmp2)-0.05, np.max(tmp2)+0.05, 0.01)

    # print(x1, x2)

    X1, X2 = np.meshgrid(x1, x2)
    
    fig = plt.figure()
    plt.title("Function Values vs x")
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    f = np.exp(X1+(3*X2)-0.1) + np.exp(X1-(3*X2)-0.1) + np.exp(-X1-0.1)
    ax.plot_surface(X1, X2, f)
    for i in range(len(tmp1)-1):
        ax.plot(tmp1[i], tmp2[i], func[i], marker='.', linestyle='None', label='Label', color='red', zorder=10)
    i = len(tmp1) - 1
    ax.plot(tmp1[i], tmp2[i], func[i], marker='.', linestyle='None', label='Label', color='yellow', zorder=10)
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.view_init(50, 100)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    ax.contour(X1, X2, f)
    for i in range(len(tmp1)-1):
        ax.plot(tmp1[i], tmp2[i], marker='.', linestyle='None', label='Label', color='red', zorder=10)
        arrow = FancyArrowPatch((tmp1[i], tmp2[i]), (tmp1[i+1], tmp2[i+1]), arrowstyle='simple', color='k', mutation_scale=10)
        ax.add_patch(arrow)
    i = len(tmp1) - 1
    ax.plot(tmp1[i], tmp2[i], marker='.', linestyle='None', label='Label', color='yellow', zorder=10)

    x1 = np.arange(np.min(tmp1)-0.05, np.max(tmp1)+0.05, 0.05)
    x2 = np.arange(np.min(tmp2)-0.05, np.max(tmp2)+0.05, 0.05)
    print(len(x1)*len(x2))
    p = 1
    print("Point No.")
    for v1 in x1:
        for v2 in x2:
            print(p, end="\r")
            for i in range(len(tmp1)):
                hessian = get_hessian(np.array([tmp1[i], tmp2[i]]))
                dir = np.array([v1, v2])
                dir_t = dir.transpose()
                if dir_t.dot(hessian.dot(dir)) <= 1:
                    ax.scatter(v1, v2, c="blue")
            p+=1

    plt.show()

