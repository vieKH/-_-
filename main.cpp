#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <omp.h>
#define MAX_SIZE 500

using namespace std;

template <class T>
void multiply_non_omp(T** matrix1, T** matrix2, T** result, int size)
{
	for (int row = 0; row < size; row++) {
		for (int col = 0; col < size; col++) {
			T sum_result = 0;
			for (int i = 0; i < size; i++) {
				sum_result += matrix1[row][i] * matrix2[i][col];
			}
			result[row][col] = sum_result;
		}
	}
}

void generate_matrix(std::string file_path)
{
	int* matrix1, * matrix2;
	matrix1 = new int[MAX_SIZE * MAX_SIZE];
	matrix2 = new int[MAX_SIZE * MAX_SIZE];

	for (int i = 0; i < MAX_SIZE * MAX_SIZE; i++) {
		matrix1[i] = (rand() % MAX_SIZE + 10);
		matrix2[i] = (rand() % MAX_SIZE - 12);
	}
	std::ofstream data(file_path);
	for (int i = 0; i < MAX_SIZE * MAX_SIZE; i++) {
		data << matrix1[i] << " " << matrix2[i] << " ";
	}
	data.close();
}

template <class T>
void read_file_to_matrix(std::string file_path, T** matrix1, T** matrix2)
{
	int temp;
	bool changer = true;
	int i1 = 0;
	int i2 = 0;
	std::ifstream data(file_path);

	while (data >> temp) {

		if (changer) {
			int row = i1 / MAX_SIZE;
			int column = i1 - row * MAX_SIZE;
			matrix1[row][column] = temp;
			i1++;
		}
		else {
			int row = i2 / MAX_SIZE;
			int column = i2 - row * MAX_SIZE;
			matrix2[row][column] = temp;
			i2++;
		}
		changer = !changer;
	}

	data.close();
}

template <class T>
void write_result_to_file(std::string file_path, T** result, int size)
{
	std::ofstream matrix(file_path);

	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; j++)
			matrix << result[i][j] << " ";
	}
	matrix.close();
}


int main() {

	int** matrix1, ** matrix2, ** result;

	matrix1 = new int* [MAX_SIZE];
	for (int i = 0; i < MAX_SIZE; i++)
		matrix1[i] = new int[MAX_SIZE];

	matrix2 = new int* [MAX_SIZE];
	for (int i = 0; i < MAX_SIZE; i++)
		matrix2[i] = new int[MAX_SIZE];

	result = new int* [MAX_SIZE];
	for (int i = 0; i < MAX_SIZE; i++)
		result[i] = new int[MAX_SIZE];

	for (int i = 0; i < MAX_SIZE; i++)
	{
		for (int j = 0; j < MAX_SIZE; j++)
		{
			matrix1[i][j] = i + j;
			matrix2[i][j] = i - j;
		}
	}

	generate_matrix("DataMatrix.txt");

	read_file_to_matrix("DataMatrix.txt", matrix1, matrix2);

	std::vector<int> sizes = { 100, 200, 300, 400, 500 };
	int s = 0;

	std::ofstream file("E:/Study/ПП/lab1/result.txt");
	while (s < (int)sizes.size()) {
		const int size = sizes[s];

		auto start1 = std::chrono::steady_clock::now();
		multiply_non_omp(matrix1, matrix2, result, size);
		auto end1 = std::chrono::steady_clock::now();
		std::cout << ">> Using non-omp: " << endl;
		std::cout << "\tMatrix's size is: " << size << "x" << size << std::endl;
		std::cout << "\tDef meth\'s time: " << std::chrono::duration<double>(end1 - start1).count() << std::endl;
		if (size == MAX_SIZE)
			write_result_to_file("Mul1.txt", result, MAX_SIZE);

		s++;
	}
	file.close();

	return 0;
}