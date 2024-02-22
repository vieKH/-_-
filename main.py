import numpy as np
MAX_SIZE = 500


def read_file_to_matrix(file_path):
    data = np.loadtxt(file_path)
    matrix1 = data[::2].reshape((MAX_SIZE, MAX_SIZE))
    matrix2 = data[1::2].reshape((MAX_SIZE, MAX_SIZE))
    return matrix1, matrix2


def multiply_matrices(matrix1, matrix2):
    return np.dot(matrix1, matrix2)


def main():
    matrix1, matrix2 = read_file_to_matrix("E:/Study/ПП/lab1/lab1/DataMatrix.txt")
    res = multiply_matrices(matrix1, matrix2)
    with open("E:/Study/ПП/lab1/lab1/Mul1.txt", "r") as f:
        data = f.read().split()
    data = np.reshape(np.asarray(data), (500, 500))

    if np.array_equal(data.astype(int), res.astype(int)):
        print("Congratulation, this test was passed!")
    else:
        print("Fuiyooo, check it again (T_T)")


if __name__ == "__main__":
    main()
