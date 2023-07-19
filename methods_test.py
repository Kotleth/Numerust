from numerust import *
import numpy as np
from scipy.linalg import lu


mat_A = [20.0, 1.0, 1.0, -1.0, 2.0, -30.0, 3.0, 1.0, -2.0, 3.0, -25.0, 5.1, 2.1, 2.0, 1.11, 27.3]
vec_b = [20.0, 70.0, -30.0, 5.0]
print(gauss_seidel(mat_A, vec_b, timeout=1000))
print(newton_optimization([2, -3, 1]))

print(matrix_inv([2.0, 3.0, 5.0, 1.0, 4.0, 6.0, 7.0, 8.0, 9.0]))