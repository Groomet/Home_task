
import functions as task

matrix_1 = task.check_time(task.m)
matrix_2 = task.check_time(task.m)
vector_1 = task.check_time(task.v)
vector_2 = task.check_time(task.v)

matrix50 = task.check_time(task.m50)
matrix50_2 = task.check_time(task.m50)
vector51 = task.check_time(task.v50)
vector52 = task.check_time(task.v50)

task.write_to_file('matrix_1.txt', matrix_1)
task.write_to_file('matrix_2.txt', matrix_2)
task.write_vector_to_file('vector_1.txt', vector_1)
task.write_vector_to_file('vector_2.txt', vector_2)

matrix1_from_file = task.read_from_file('matrix_1.txt')
matrix2_from_file = task.read_from_file('matrix_2.txt')
vector1_from_file = task.read_vector_from_file('vector_1.txt')
vector2_from_file = task.read_vector_from_file('vector_2.txt')

print("Матрица1:")
for row in matrix_1:
    print(row)
print("\nМатрица 2:")
for row in matrix_2:
    print(row)

result_matrix = task.check_time(task.mm, matrix_1, matrix_2)

print("\nРезультат умножения матрицы 1 на матрицу 2:")
for row in result_matrix:
    print(row)

print("\nВектор1:")
print(vector_1)
print("\nВектор2:")
print(vector_2)

result_vector = task.check_time(task.vm, matrix_1, vector_1)
print("Результат умножения матрицы 1 на вектор:", result_vector)

trace_matrix1 = task.matrix_trace(matrix_1)
print("\nСлед матрицы 1:", trace_matrix1)

dot_product = task.vector_dot_product(vector_1, vector_2)
print("\nСкалярное произведение векторов:", dot_product)

num_bins = 4
histogram, bin_edges = task.calculate_histogram(vector_1, num_bins)
print("\nГистограмма для вектора 1 (с {} квантами):".format(num_bins))
print("Частоты:", histogram)
print("Границы интервалов:", bin_edges)

kernel = [-1, 0, 1]
filtered_vector = task.kernel_filter(vector_1, kernel)

print("\nРезультат фильтрации вектора ядерным фильтром:")
print(filtered_vector)

task.check_time(task.mm50, matrix50, matrix50_2)
task.check_time(task.vm50, matrix50, vector51)
