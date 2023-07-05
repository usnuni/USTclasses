
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.autograd.functional import hessian
from sympy import *

# 함수
def f(x, y):
    # f(x, y) = (x+y)(xy+xy^2)
    return (x + y) * (x * y + x * y * y)

# x, y 구간 3d 그래프 도사
x = np.linspace(-1, 1.5, 100)
y = np.linspace(-1.2, 0.2, 100)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('f(x, y) = (x+y)(xy+xy^2)')
plt.show()

# Gradiant

def gradiant(x,y) :
    # f(x, y) = (x+y)(xy+xy^2)
    # df/dx = y + 2xy + y^2
    # df/dy = x + xy + x^2
    return np.array([y + 2*x*y + y*y, x + x*y + x*x])

p = np.array([1, 0])
print(gradiant(p[0], p[1]))

grad = gradiant(p[0], p[1])
grad = grad / np.linalg.norm(grad)

plt.contour(X, Y, Z, 50, cmap='viridis')
plt.quiver(p[0], p[1], grad[0], grad[1], color='red', scale=10)
plt.title("Gradient 2d")
plt.show()

x = np.ones((100))
y = np.linspace(-1.2, 0.2, 100)
z = y
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=2,cstride=2,cmap='viridis',linewidth=0.5,antialiased=True)
ax.plot(x,y,z, c='g')
ax.scatter([1],[0],[0], c='r', s=50)
fig.colorbar(surf,shrink=0.5,aspect=5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Gradient 3d")
plt.show()




# critial point

def gradient_to_zero(symbols_list, partials):
    # find critical point
    # grad = 0
    # grad = [df/dx, df/dy]
    # grad = [0, 0]
    # x = 0, y = 0
    partial_x = Eq(partials[0], 0)
    partial_y = Eq(partials[1], 0)

    sol = solve((partial_x, partial_y), (symbols_list[0], symbols_list[1]))
    print("Singular point is : {0}".format(sol))
    return sol

def partial(element, function):
    partial_diff = function.diff(element)

    return partial_diff

def get_critical_points():

    x, y = symbols('x y', real=True)
    symbols_list = [x, y]
    function = (x + y) * (x * y + x * y * y)
    partials, partials_second = [], []

    for element in symbols_list:
        partial_diff = partial(element, function)
        partials.append(partial_diff)

    singular = gradient_to_zero(symbols_list, partials)
    return singular

critical_points = get_critical_points()
critical_points = np.array(critical_points, dtype=np.float64)
print("cirtical points : ", critical_points)
plt.contour(X, Y, Z, 50, cmap='viridis')
plt.scatter(critical_points[:, 0], critical_points[:, 1], color='red', s=50)
plt.title("Critical Points")
plt.show()


# Hessian

def hessian(x, y):
    # f(x, y) = (x+y)(xy+xy^2)
    # df/dx = y + 2xy + y^2
    # df/dy = x + xy + x^2
    # d2f/dx2 = 2 * y * (y + 1)
    # d2f/dy2 = 2 * x * (x + 3 * y + 1)
    # d2f/dxdy = x * (4 * y + 2) + y * (3 * y + 2)
    return np.array([[2 * y * (y + 1), x * (4 * y + 2) + y * (3 * y + 2)], [x * (4 * y + 2) + y * (3 * y + 2), 2 * x * (x + 3 * y + 1)]])

for x, y in critical_points:
    hess = hessian(x, y)
    #print(np.linalg.eigvals(hess))

    if np.all(np.linalg.eigvals(hess) > 0):
        print("x: {}, y: {} is a local minimum".format(x, y))
    elif np.all(np.linalg.eigvals(hess) < 0):
        print("x: {}, y: {} is a local maximum".format(x, y))
    else:
        print("x: {}, y: {} is a saddle point".format(x, y))

plt.contour(X, Y, Z, levels=200, cmap="viridis")

for x, y in critical_points:
    print(f"Critical point: {x, y}")
    hess = hessian(x, y)
    print(f"critical point: {np.array([x, y])}, hessian: {hess}")
    eigenvalues, eigenvectors = np.linalg.eig(hess)
    print(f"eigenvalues: {eigenvalues}, eigenvectors: {eigenvectors}")

    if eigenvalues[0] > 0 and eigenvalues[1] > 0:
        print(f"local minimum: {np.array([x, y])}")
        color = "red"

    elif eigenvalues[0] < 0 and eigenvalues[1] < 0:
        print(f"local maximum: {np.array([x, y])}")
        color = "blue"

    else:
        print(f"saddle point: {np.array([x, y])}")
        color = "green"
    for i in range(eigenvectors.shape[1]):
        plt.scatter(x, y, color=color)
        plt.quiver(x, y, eigenvectors[0][i], eigenvectors[1][i], color=color, scale=10)
plt.show()