from numerust import *
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv
import time
from scipy.linalg import lu

def test_least_sq_approx(x, y, degree):
    coefs = list(l_square_approx(x, y, degree + 1))
    x_values = np.linspace(min(some_x)-5, max(some_x)+5, 100)  # 100 points between -10 and 10

    for i in range(4 - len(coefs)):
        coefs.append(0)
    print(coefs)
    # Evaluate the function for each x-value
    y_values = coefs[0] + coefs[1] * x_values + coefs[2] * x_values ** 2 + coefs[3] * x_values ** 3
    plt.plot(x_values, y_values)
    plt.plot(some_x, some_y, 'o')
    plt.show()
    # return x

if __name__ == "__main__":
    print(np.linalg.inv([[9, 90, 1140], [90, 1140, 16200], [1140, 16200, 245328]]))

    some_x = [2.0, 4.0, 6.0, 8.0, 10.0]
    some_y = []
    # print(some_x)
    for x in some_x:
        some_y.append(2 - 4*x - 2*x**2 + x**3)
    print(some_y)
    test_least_sq_approx(some_x, some_y, 3)


# matrix = [[2.0, 3.0, 5.0, 6.0], [1.0, 4.0, 6.0, 9.0], [7.0, 8.0, 9.0, 11.0], [3.0, -4.0, 7.0, 13.0]]
# print(matrix)
# # mat_A = [20.0, 1.0, 1.0, -1.0, 2.0, -30.0, 3.0, 1.0, -2.0, 3.0, -25.0, 5.1, 2.1, 2.0, 1.11, 27.3]
# # vec_b = [20.0, 70.0, -30.0, 5.0]
# # print(gauss_seidel(mat_A, vec_b, timeout=1000))
# # print(newton_optimization([2, -3, 1]))
# my_list = []
# for i in matrix:
#     for n in i:
#         my_list.append(n)
# print(my_list)
#
# start = time.time()
# # print(matrix_inv([2.0, 3.0, 5.0, 1.0, 4.0, 6.0, 7.0, 8.0, 9.0]))
# my_res = matrix_inv(my_list)
# my_end = time.time()
# print(my_res)
# print(my_end - start)
#
# start = time.time()
# print(inv(matrix))
# my_end = time.time()
# print(my_end - start)