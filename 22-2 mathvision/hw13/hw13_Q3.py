import numpy as np
import matplotlib.pyplot as plt
import torch

def f(x, y):
    # f(x, y) = sin(x+y+1) + (x-y-1)^2 - 1.5x + 2.5y + 1
    return np.sin(x+y+1) + (x-y-1)**2 - 1.5*x + 2.5*y + 1

x = np.linspace(-1, 5, 100)
y = np.linspace(-3, 4, 100)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('f(x, y) = sin(x+y+1) + (x-y-1)^2 - 1.5x + 2.5y + 1')
plt.show()

def gradient(x, y):
    return np.array([np.cos(x+y+1) + 2*(x-y-1) - 1.5, np.cos(x+y+1) - 2*(x-y-1) + 2.5])

plt.contour(X, Y, Z, levels=200, cmap="viridis")
grad = gradient(X, Y)
plt.quiver(X, Y, grad[0], grad[1], color="red", scale=20)
plt.show()


# gradient descent method

# def gradient_descent(x, y, lr, n_iter):

#     x_list = [x]
#     y_list = [y]
#     z_list = [f(x, y)]
#     for i in range(n_iter):
#         grad = gradient(x, y)
#         x -= lr * grad[0]
#         y -= lr * grad[1]
#         x_list.append(x)
#         y_list.append(y)
#         z = f(x, y)
#         z_list.append(z)


def gradient_descent(start_point, lr=0.01, n_iter=2000):
    """_summary_

    Args:
        start_point (_type_): _description_
        lr (float, optional): _description_. Defaults to 0.01.
        n_iter (int, optional): _description_. Defaults to 1000.
    """
    # gradient descent until convergence
    print(f"gradient descent start point {start_point}")
    pathway = []
    p = np.array(start_point)
    prep = p.copy()
    for i in range(n_iter):
        grad = gradient(p[0], p[1])
        p -= lr * grad
        if i % 5 == 0:
            pathway.append(p.copy())
        if np.linalg.norm(p - prep) < 1e-8:
            print(f"gradient descent converge at {i} steps")
            break
        prep = p.copy()
    pathway = np.array(pathway)
    # print(f"gradient descent: {pathway}")
    print(f"gradient descent converge point {pathway[-1]}")

    plt.contour(X, Y, Z, levels=200, cmap="viridis", alpha=0.5)
    plt.scatter(pathway[0, 0], pathway[0, 1], color="red")
    plt.scatter(pathway[-1, 0], pathway[-1, 1], color="red")
    plt.quiver(pathway[:-1, 0], pathway[:-1, 1], pathway[1:, 0] - pathway[:-1, 0], pathway[1:, 1] - pathway[:-1, 1], color="green", scale=10)
    plt.show()

    # fig = plt.figure()
    # ax = fig.gca(projection='3d') #viridis 하면 잘 안보임
    # ax.plot_surface(X, Y, Z, cmap='RdBu_r', edgecolor='none')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.set_title('f(x, y) = sin(x+y+1) + (x-y-1)^2 - 1.5x + 2.5y + 1')
    # ax.plot(pathway[:, 0], pathway[:, 1], f(pathway[:, 0], pathway[:, 1]), color="red")
    # plt.show()


gradient_descent([0.0, 0.0])
gradient_descent([4.0, 3.0])
gradient_descent([1.0, 1.0])
gradient_descent([3.0, -3.0])


# newton's method

def hessian(x, y):
    return np.array([[np.cos(x+y+1) + 2, -2], [-2, np.cos(x+y+1) + 2]])

def newton_method(start_point, lr=0.01, n_iter=2000):
    """_summary_

    Args:
        start_point (_type_): _description_
        lr (float, optional): _description_. Defaults to 0.01.
        n_iter (int, optional): _description_. Defaults to 1000.
    """
    print(f"newton's method start point {start_point}")
    pathway = []
    p = np.array(start_point)
    prep = p.copy()
    for i in range(n_iter):
        grad = gradient(p[0], p[1])
        hess = hessian(p[0], p[1])
        p -= lr * np.linalg.inv(hess) @ grad
        if i % 5 == 0:
            pathway.append(p.copy())
        if np.linalg.norm(p - prep) < 1e-8:
            print(f"newton's method converge at {i} steps")
            break
        prep = p.copy()
    pathway = np.array(pathway)
    print(f"newton's method converge point {pathway[-1]}")

    plt.contour(X, Y, Z, levels=200, cmap="viridis", alpha=0.5)
    plt.scatter(pathway[0, 0], pathway[0, 1], color="red")
    plt.scatter(pathway[-1, 0], pathway[-1, 1], color="red")
    plt.quiver(pathway[:-1, 0], pathway[:-1, 1], pathway[1:, 0] - pathway[:-1, 0], pathway[1:, 1] - pathway[:-1, 1], color="blue", scale=10)
    plt.show()

    # fig = plt.figure()
    # ax = fig.gca(projection='3d') #viridis 하면 잘 안보임
    # ax.plot_surface(X, Y, Z, cmap='RdBu_r', edgecolor='none')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.set_title('f(x, y) = sin(x+y+1) + (x-y-1)^2 - 1.5x + 2.5y + 1')
    # ax.plot(pathway[:, 0], pathway

newton_method([0.0, 0.0])
newton_method([4.0, 3.0])
newton_method([1.0, 1.0])
newton_method([3.0, -3.0])
