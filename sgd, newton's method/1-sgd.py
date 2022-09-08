# steepest gradient descent to calculate direction of descent
# and for calculating the step size to take in that direction, we use
# 1. with exact line search
# 2. with backtracking line search

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
    
def get_stepsize(type, x_current, dir, init_alpha):
    tau = 0.7
    beta = 0.1
    alpha = init_alpha
    if type == "armijo-goldstein":
        current_grad = get_gradient(x_current)
        current_f = function(x_current)
        x_next = x_current + alpha * (dir)
        while True:
            if np.isnan(function(x_next)):
                alpha *= tau
            else:
                if function(x_next) <= current_f + (1-beta) * current_grad.dot(x_next - x_current) or function(x_next) >= current_f + beta * current_grad.dot(x_next - x_current):
                    alpha *= tau
                else:
                    break
            x_next = x_current + alpha * (dir)
    elif type == "backtracking":
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


def solve(initial_x, typ):
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

    if len(sys.argv) > 1:
        typ = str(sys.argv[1])

    while True:
        if iteration >= max_iter:
            print("Maximum iteration exceeded.")
            break
        current_grad = get_gradient(x_current)
        if np.linalg.norm(current_grad) < eps:
            print("Required tolerance is achieved.")
            break
        alpha = get_stepsize(typ, x_current, -current_grad, init_alpha)
        print(alpha)
        x_next = x_current - alpha * current_grad
        x_vals.append(x_next)
        f_vals.append(function(x_next))
        x_current = x_next
        iteration += 1

    print("Method used: Steepest Gradient Descent with {} line search".format(typ))
    print("Value of x1, x2: {}, {}".format(x_current[0], x_current[1]))
    print("Number of iterations: {}".format(iteration))
    print("Function value: {}".format(function(x_current)))
    return x_vals, f_vals


if __name__== "__main__":
    x_0 = np.array([0, 0])
    x, func = solve(x_0, "armijo-goldstein")

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
    plt.show()


    print()
    print()

    x, func = solve(x_0, "backtracking")

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
    plt.show()



