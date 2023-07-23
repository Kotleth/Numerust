from numerust import *
import numpy as np

from numpy.linalg import inv
import time
from scipy.linalg import lu

matrix = [[2.0, 3.0, 5.0, 6.0], [1.0, 4.0, 6.0, 9.0], [7.0, 8.0, 9.0, 11.0], [3.0, -4.0, 7.0, 13.0]]
print(matrix)
# mat_A = [20.0, 1.0, 1.0, -1.0, 2.0, -30.0, 3.0, 1.0, -2.0, 3.0, -25.0, 5.1, 2.1, 2.0, 1.11, 27.3]
# vec_b = [20.0, 70.0, -30.0, 5.0]
# print(gauss_seidel(mat_A, vec_b, timeout=1000))
# print(newton_optimization([2, -3, 1]))
my_list = []
for i in matrix:
    for n in i:
        my_list.append(n)
print(my_list)

start = time.time()
# print(matrix_inv([2.0, 3.0, 5.0, 1.0, 4.0, 6.0, 7.0, 8.0, 9.0]))
my_res = matrix_inv(my_list)
my_end = time.time()
print(my_res)
print(my_end - start)

start = time.time()
print(inv(matrix))
my_end = time.time()
print(my_end - start)