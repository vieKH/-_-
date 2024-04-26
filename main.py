import numpy as np

matrixA = np.loadtxt('matrixA.txt')
matrixB = np.loadtxt('matrixB.txt')
result = np.loadtxt('matrix_result.txt')

result_calculated = np.dot(matrixA.reshape(1000, 1000), matrixB.reshape(1000, 1000))

if np.allclose(result.reshape(1000, 1000), result_calculated):
    print("Congratulation! Result is exactly\n")
else:
    print("No no no, check it again \n")