import ctypes
import os
import numpy as np

# Wczytaj bibliotekę dynamiczną

new_path = os.getcwd()
my_lib = ctypes.CDLL(f'{new_path}/target/release/libtesting.dylib')

# Zdefiniuj argumenty i typ zwracany przez funkcję z biblioteki dynamicznej
# my_lib.newton_optimisation_polynomial.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float]
my_lib.newton_optimisation_polynomial.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int, ctypes.c_float, ctypes.c_float]
my_lib.newton_optimisation_polynomial.restype = ctypes.c_float


# x_multipliers = [1.0, -7.0]
# data_vec = np.array(x_multipliers, dtype=np.float32)
# # x_multipliers_array = (ctypes.c_float * len(x_multipliers))(*x_multipliers)
# x_zero = 2.0
# error = 0.00000001
# # Wywołaj funkcję z biblioteki dynamicznej
# result = my_lib.newton_optimisation_polynomial(data_vec, len(x_multipliers), x_zero, error)
#
# # Wyświetl wynik
# print(result)


# my_lib.vis_mat.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_size_t]
# my_lib.vis_mat.restype = None
#
def vis_matrix(array):
    my_lib.vis_mat.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t]
    my_lib.vis_mat.restype = ctypes.c_float


    x_multipliers = [1.0, -7.0, 2.0, 11.0]
    data_vec = np.array(x_multipliers, dtype=np.float32)
    # Wywołaj funkcję z biblioteki dynamicznej
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


def gauss_seidel(b_vector):

    my_lib.gauss_seidel.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int, np.ctypeslib.ndpointer(dtype=np.float32), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    my_lib.gauss_seidel.restype = ctypes.POINTER(ctypes.c_float)


    a_matrix = np.array([20.0, 1.0, 1.0, -1.0, 2.0, -30.0, 3.0, 1.0, -2.0, 3.0, -25.0, 5.1, 2.1, 2.0, 1.11, 27.3], dtype=np.float32)
    b_vector = np.array([20.0, 70.0, -30.0, 5.0], dtype=np.float32)
    # a_matrix = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    # b_vector = np.array([11.0, 13.0], dtype=np.float32)

    # Wywołaj funkcję z biblioteki dynamicznej
    result_ptr = my_lib.gauss_seidel(a_matrix, len(a_matrix), b_vector, len(b_vector), int(np.sqrt(len(a_matrix))), int(np.sqrt(len(a_matrix))))
    result_array = np.ctypeslib.as_array(result_ptr, shape=(len(b_vector),))
    x = result_array.copy()
    my_lib.free(result_ptr)
    ###
    # correct = [1.0754, -2.1643, 0.89972, 0.22248]
    # _equ = 0
    # for _z in range(len(b_vector)):
    #     _equ += a_matrix[_z]*correct[_z]
    # print(_equ - 20)
    A = list(a_matrix)
    b = list(b_vector)
    new_x = list(x)
    print(x)
    for i in range(len(b_vector)):
        equ = 0
        for j in range(len(b_vector)):
            equ += A[i*len(b_vector) + j]*new_x[j]
        print(f"Now {equ}, should be {b[i]} so delta is {equ - b[i]}")


gauss_seidel(2)

# vis_matrix([1])

# xyz = np.array([[1, 0, 0], [4, 0.2, 0], [7, 8, 1/9]])
# print(np.linalg.inv(xyz))