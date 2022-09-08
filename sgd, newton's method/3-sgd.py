# steepest descent with step size alpha
# 1. 0 < alpha < 2/lambda_max
# 2. alpha > 2/lambda_max

# x = (x1, x2)
# f(x1, x2) = 5*((x1)^2) + 5*((x2)^2) - (x1*x2) - (11*x1) + (11*x2) + 11

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

def function(x):
    x1 = x[0]
    x2 = x[1]
    y = 5*(pow(x1,2)) + 5*(pow(x2,2)) - (x1*x2) - (11*x1) + (11*x2) + 11
    return y


def get_gradient(x):
    x1 = x[0]
    x2 = x[1]
    dydx1 = (10*x1)-x2-11
    dydx2 = (10*x2)-x1+11
    return np.array([dydx1, dydx2])


def get_hessian_max_eig(x):
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
  w, v = np.linalg.eig(hessian)
  return np.max(w)


def solve_part1(initial_x):
    eps = 1e-6
    init_alpha = 1
    max_iter = 100

    x_vals = []
    f_vals = []
    x_current = initial_x.copy()

    iteration = 0
    current_grad = None

    x_vals.append(initial_x)
    f_vals.append(function(initial_x))

    alpha = None

    while True:
        if iteration >= max_iter:
            print("Maximum iteration exceeded.")
            break
        current_grad = get_gradient(x_current)
        if np.linalg.norm(current_grad) < eps:
            print("Required tolerance is achieved.")
            break
        if alpha == None:
            alpha = (2/get_hessian_max_eig(x_current)) - (0.1)
    
        x_next = x_current - alpha * current_grad
        x_vals.append(x_next)
        f_vals.append(function(x_next))
        x_current = x_next
        iteration += 1

    print("Method used: Steepest Descent Method with {} step size".format(alpha))
    print("Value of x1, x2: {}, {}".format(x_current[0], x_current[1]))
    print("Number of iterations: {}".format(iteration))
    print("Function value: {}".format(function(x_current)))
    return x_vals, f_vals

def solve_part2(initial_x):
    eps = 1e-6
    init_alpha = 1
    max_iter = 100

    x_vals = []
    f_vals = []
    x_current = initial_x.copy()

    iteration = 0
    current_grad = None

    x_vals.append(initial_x)
    f_vals.append(function(initial_x))

    alpha = None

    while True:
        if iteration >= max_iter:
            print("Maximum iteration exceeded.")
            break
        current_grad = get_gradient(x_current)
        if np.linalg.norm(current_grad) < eps:
            print("Required tolerance is achieved.")
            break
        if alpha == None:
            alpha = (2/get_hessian_max_eig(x_current)) + (0.1)
    
        x_next = x_current - alpha * current_grad
        x_vals.append(x_next)
        f_vals.append(function(x_next))
        x_current = x_next
        iteration += 1

    print("Method used: Steepest Descent Method with {} step size".format(alpha))
    print("Value of x1, x2: {}, {}".format(x_current[0], x_current[1]))
    print("Number of iterations: {}".format(iteration))
    print("Function value: {}".format(function(x_current)))
    return x_vals, f_vals


if __name__== "__main__":
    x_0 = np.array([0.5, 0.5])

    print()
    print("Part 1")
    print()

    x, func = solve_part1(x_0)

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
    f = 5*(pow(X1,2)) + 5*(pow(X2,2)) - (X1*X2) - (11*X1) + (11*X2) + 11
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
    plt.show()

    print()
    print("Part 2")
    print()

    x, func = solve_part2(x_0)

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
    f = 5*(pow(X1,2)) + 5*(pow(X2,2)) - (X1*X2) - (11*X1) + (11*X2) + 11
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
    plt.show()



