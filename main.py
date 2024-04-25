import numpy as np

matrixA = np.loadtxt('matrixA.txt')
matrixB = np.loadtxt('matrixB.txt')
result = np.loadtxt('matrix_result.txt')

result_calculated = np.dot(matrixA, matrixB)

if np.allclose(result, result_calculated):
    print("Congratulation! Result is exactly\n")
else:
    print("No no no, check it again \n")