import ctypes
from os import getcwd
import numpy as np
new_path = getcwd()
my_lib = ctypes.CDLL(f'{new_path}/target/release/libtesting.dylib')


'''
Numerust Wrapper
'''

def newton_optimization(coefficients, x_start=1.0, eps=1e-6):
    my_lib.newton_optimisation_polynomial.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int, ctypes.c_float, ctypes.c_float]
    my_lib.newton_optimisation_polynomial.restype = ctypes.c_float
    x_multipliers = coefficients
    data_vec = np.array(x_multipliers, dtype=np.float32)
    x_zero = x_start
    error = eps
    result = my_lib.newton_optimisation_polynomial(data_vec, len(x_multipliers), x_zero, error)

    return result

def vis_matrix(array):
    my_lib.vis_mat.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t]
    my_lib.vis_mat.restype = ctypes.c_float

    x_multipliers = [1.0, -7.0, 2.0, 11.0]
    data_vec = np.array(x_multipliers, dtype=np.float32)
    my_lib.vis_mat(data_vec, len(x_multipliers), 2, 2)

# def vis_matrix(array):
#     # Przykładowe dane wejściowe
#
#     my_lib.vis_mat.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32)]
#     my_lib.vis_mat.restype = ctypes.c_float
#
#     x_multipliers = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
#     data_vec = np.array(x_multipliers, dtype=np.float32)
#     # x_multipliers_array = (ctypes.c_float * len(x_multipliers))(*x_multipliers)s
#     # Wywołaj funkcję z biblioteki dynamicznej
#     my_lib.vis_mat(data_vec, len(x_multipliers))


def gauss_seidel(mat_a, vec_b, timeout=500):
    my_lib.gauss_seidel.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    my_lib.gauss_seidel.restype = ctypes.POINTER(ctypes.c_float)

    a_matrix = np.array(mat_a, dtype=np.float32)
    b_vector = np.array(vec_b, dtype=np.float32)

    result_ptr = my_lib.gauss_seidel(a_matrix, len(a_matrix), b_vector, len(b_vector), int(np.sqrt(len(a_matrix))), timeout)
    result_array = np.ctypeslib.as_array(result_ptr, shape=(len(b_vector),))
    x = result_array.copy()
    my_lib.free(result_ptr)
    return x


def matrix_inv(mat_a):
    my_lib.matrix_inv.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int, ctypes.c_int]
    my_lib.matrix_inv.restype = ctypes.POINTER(ctypes.c_float)

    a_matrix = np.array(mat_a, dtype=np.float32)
    n = int(np.sqrt(len(a_matrix)))

    result_ptr = my_lib.matrix_inv(a_matrix, len(mat_a), n)
    result_array = np.ctypeslib.as_array(result_ptr, shape=(len(mat_a),))
    x = result_array.copy()
    my_lib.free(result_ptr)
    temp_mat = []
    for i in range(n):
        temp_mat.append([])
        for j in range(n):
            temp_mat[i].append(x[i*n + j])
    return np.array(temp_mat)