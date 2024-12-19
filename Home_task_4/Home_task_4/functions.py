import random
from time import time
import numpy as np

def v():
    return [random.uniform(0, 1) for _ in range(5)]
vector = v()
def m():
    return [[random.uniform(0, 1) for _ in range(5)] for _ in range(5)]
matrix1 = m()

def vm(matrix1, vector):
    return [sum(matrix1[i][j] * vector[j] for j in range(5)) for i in range(5)]
s=vm(matrix1, vector)

def mm(matrix1, matrix2):
    result = [[0 for _ in range(5)] for _ in range(5)]
    for i in range(5):
        for j in range(5):
            result[i][j] = sum(matrix1[i][k] * matrix2[k][j] for k in range(5))

    return result

def matrix_trace(matrix):
    return sum(matrix[i][i] for i in range(5))

def vector_dot_product(vector1, vector2):
    return sum(vector1[i] * vector2[i] for i in range(len(vector1)))

def calculate_histogram(vector, num_bins):
    histogram, bin_edges = np.histogram(vector, bins=num_bins)
    return histogram, bin_edges



def kernel_filter(vector, kernel):
    filtered_vector = [0] * len(vector)
    kernel_size = len(kernel)
    padding = kernel_size // 2
    for i in range(padding, len(vector) - padding):
        filtered_vector[i] = sum(vector[i + j - padding] * kernel[j] for j in range(kernel_size))

    return filtered_vector

def v50():
    return [random.uniform(0, 1) for _ in range(50)]
vector50 = v50()
def m50():
    return [[random.uniform(0, 1) for _ in range(50)] for _ in range(50)]
matrix50 = m50()

def vm50(matrix50, vector50):
    return [sum(matrix50[i][j] * vector50[j] for j in range(50)) for i in range(50)]
s=vm50(matrix50, vector50)

def mm50(matrix50, matrix52):
    result = [[0 for _ in range(50)] for _ in range(50)]
    for i in range(50):
        for j in range(50):
            result[i][j] = sum(matrix50[i][k] * matrix52[k][j] for k in range(50))

    return result

def matrix_trace50(matrix):
    return sum(matrix[i][i] for i in range(50))

def vector_dot_product(vector51, vector52):
    return sum(vector51[i] * vector52[i] for i in range(len(vector51)))

def calculate_histogram(vector50, num_bins):
    histogram, bin_edges = np.histogram(vector50, bins=num_bins)
    return histogram, bin_edges


def kernel_filter(vector50, kernel):
    filtered_vector50 = [0] * len(vector50)
    kernel_size = len(kernel)
    padding = kernel_size // 2
    for i in range(padding, len(vector50) - padding):
        filtered_vector50[i] = sum(vector50[i + j - padding] * kernel[j] for j in range(kernel_size))

    return filtered_vector50

def write_to_file(filename, data):
    with open(filename, 'w') as f:
        for row in data:
            f.write(' '.join(map(str, row)) + '\n')

def read_from_file(filename):
    with open(filename, 'r') as f:
        data = [list(map(float, line.split())) for line in f]
    return data

def write_vector_to_file(filename, vector):
    with open(filename, 'w') as f:
        f.write(' '.join(map(str, vector)) + '\n')

def read_vector_from_file(filename):
    with open(filename, 'r') as f:
        return list(map(float, f.readline().split()))


def check_time(func, *args, fname='file.txt'):
    t1 = time()
    res = func(*args)
    t = time() - t1

    with open(fname, 'a') as f:
        print(f'{func.__name__}: {t}', file=f)

    return res
