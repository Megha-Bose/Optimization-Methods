# checking whether x* is local maxima/minima/saddle point

# x = (x1, x2)
# f1(x1, x2) = (10*(x1^2)) + (10*(x1*x2)) + (x2^2) + (4*x1) - (10*x2) + 2
# f2(x1, x2) = (16*(x1^2)) + (8*x1*x2) + (10*(x2^2)) + (12*x1) - (6*x2) + 2

import sys
import math
from markupsafe import functools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

def function1(x):
    x1 = x[0]
    x2 = x[1]
    y = (10*pow(x1,2)) + (10*(x1*x2)) + pow(x2,2) + (4*x1) - (10*x2) + 2
    return y


def function2(x):
    x1 = x[0]
    x2 = x[1]
    y = (16*pow(x1,2)) + (8*x1*x2) + (10*pow(x2,2)) + (12*x1) - (6*x2) + 2
    return y


def get_gradient1(x):
    x1 = x[0]
    x2 = x[1]
    dydx1 = (20*x1)+(10*x2)+4
    dydx2 = (10*x1)+(2*x2)-10
    return np.array([dydx1, dydx2])

def get_gradient2(x):
    x1 = x[0]
    x2 = x[1]
    dydx1 = (32*x1)+(8*x2)+12
    dydx2 = (8*x1)+(20*x2)-6
    return np.array([dydx1, dydx2])


def get_hessian_eig1(x):
  N = x.shape[0]
  hessian = np.array([[20, 10], [10, 2]])
  w, v = np.linalg.eig(hessian)
  return w

def get_hessian_eig2(x):
  N = x.shape[0]
  hessian = np.array([[32, 8],  [8, 20]])
  w, v = np.linalg.eig(hessian)
  return w


def get_dir(theta):
    return np.array([np.cos(theta), np.sin(theta)])


if __name__== "__main__":
    x_0 = np.array([0.5, 0.5])

    print()
    print("Part 1")
    print()
    
    x1 = np.arange(-5, 5, 0.1)
    x2 = np.arange(-5, 5, 0.1)

    # print(x1, x2)

    X1, X2 = np.meshgrid(x1, x2)
    
    fig = plt.figure()
    plt.title("Function Values vs x")
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    f = (10*pow(X1,2)) + (10*(X1*X2)) + pow(X2,2) + (4*X1) - (10*X2) + 2
    ax.plot_surface(X1, X2, f)
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.view_init(50, 100)
    plt.show()

    x_dash = np.array([1.8, -4])
    alpha = 0.01

    plt.title("[f(x∗ + αd_θ) − f (x∗)] vs θ")
    func_vals = []
    thetas = np.arange(0, 2*np.pi, np.pi/100)

    for theta in thetas:
        func_vals.append()
        func_vals.append(function1(x_dash+(alpha*get_dir(theta)))-function1(x_dash))
    
    plt.plot(thetas, func_vals)
    plt.show()

    print("Gradient is {}".format(get_gradient1(x_dash)))
    print("Eigen values of Hessian Matrix are {}".format(get_hessian_eig1(x_dash)))
    print("Saddle point as plotted curve gives both non negative and non positive values.")


    print()
    print("Part 2")
    print()

    x1 = np.arange(-5, 5, 0.1)
    x2 = np.arange(-5, 5, 0.1)

    # print(x1, x2)

    X1, X2 = np.meshgrid(x1, x2)
    
    fig = plt.figure()
    plt.title("Function Values vs x")
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    f = (16*pow(X1,2)) + (8*X1*X2) + (10*pow(X2,2)) + (12*X1) - (6*X2) + 2
    ax.plot_surface(X1, X2, f)
    plt.xlabel('x1')
    plt.ylabel('x2')
    ax.view_init(50, 100)
    plt.show()

    x_dash = np.array([-0.5, 0.5])
    alpha = 0.01
    plt.title("[f(x∗ + αd_θ) − f (x∗)] vs θ")
    func_vals = []
    thetas = np.arange(0, 2*np.pi, np.pi/100)

    for theta in thetas:
        func_vals.append(function2(x_dash+(alpha*get_dir(theta)))-function2(x_dash))

    plt.plot(thetas, func_vals)
    plt.show()

    print("Gradient is {}".format(get_gradient2(x_dash)))
    print("Eigen values of Hessian Matrix are {}".format(get_hessian_eig2(x_dash)))
    print("Local minima as plotted curve always non negative.")