#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <cmath>
#include <omp.h>
#define MAX_SIZE 1000

using namespace std;

struct statistical_para {
	double std;
	double mean;
	double confi_inter_left;
	double confi_inter_right;
};

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
template <class T>
void multiply_omp(T** matrix1, T** matrix2, T** result, int size, int num_thread)
{
# pragma omp parallel for num_threads(num_thread)
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
		{
			result[i][j] = 0;
			for (int k = 0; k < size; k++)
			{
				result[i][j] += matrix1[i][k] * matrix2[k][j];
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

template <class T>
statistical_para count_statistic(T** matrix, int size) {
	statistical_para ret{};
	T sum = 0;
	double sum_sq = 0;
	
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			sum += matrix[i][j];
		}
	}
	ret.mean = (double)sum / (size * size);
	
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			sum_sq += ((double)matrix[i][j] - ret.mean) * ((double)matrix[i][j] - ret.mean);
		}
	}
	ret.std = sqrt(sum_sq);

	ret.confi_inter_left = ret.mean - 1.96 * (ret.std / sqrt(size));
	ret.confi_inter_right = ret.mean + 1.96 * (ret.std / sqrt(size));

	return ret;
}


int main() {
	printf("Max thread is: %d\n", omp_get_max_threads());

	statistical_para st;
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

	generate_matrix("DataMatrix.txt");

	read_file_to_matrix("DataMatrix.txt", matrix1, matrix2);

	std::vector<int> sizes = { 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };
	int s = 0;
	
	ofstream file("Result.txt");

	st = count_statistic(matrix1, 1000);
	file << ">> Statistical parameters for matrix 1" << endl;
	cout << ">> Statistical parameters for matrix 1" << endl;
	file << "\tMean of matrix: " << st.mean << endl;
	file << "\tStandard deviation of matrix: " << st.std << endl;
	file << "\tConfidence interval with 95% confidence: (" << st.confi_inter_left << ", " << st.confi_inter_right << ")" << endl;
	cout << "\tMean of matrix: " << st.mean << endl;
	cout << "\tStandard deviation of matrix: " << st.std << endl;
	cout << "\tConfidence interval with 95% confidence: (" << st.confi_inter_left << ", " << st.confi_inter_right << ")" << endl;

	st = count_statistic(matrix2, 1000);
	file << ">> Statistical parameters for matrix 2" << endl;
	cout << ">> Statistical parameters for matrix 2" << endl;
	file << "\tMean of matrix: " << st.mean << endl;
	file << "\tStandard deviation of matrix: " << st.std << endl;
	file << "\tConfidence interval with 95% confidence: (" << st.confi_inter_left << ", " << st.confi_inter_right << ")" << endl;
	cout << "\tMean of matrix: " << st.mean << endl;
	cout << "\tStandard deviation of matrix: " << st.std << endl;
	cout << "\tConfidence interval with 95% confidence: (" << st.confi_inter_left << ", " << st.confi_inter_right << ")" << endl;
	
	file << "_______________________________" << endl;
	while (s < (int)sizes.size()) {
		const int size = sizes[s];

		auto start1 = std::chrono::steady_clock::now();
		multiply_non_omp(matrix1, matrix2, result, size);
		auto end1 = std::chrono::steady_clock::now();
		cout << ">> Using non-omp: " << endl;
		file << ">> Using non-omp: " << endl;
		cout << "\tMatrix's size is: " << size << "x" << size << std::endl;
		cout << "\tDef meth\'s time: " << std::chrono::duration<double>(end1 - start1).count() << std::endl;
		file << "\tMatrix's size is: " << size << "x" << size << std::endl;
		file << "\tDef meth\'s time: " << std::chrono::duration<double>(end1 - start1).count() << std::endl;

		auto start2 = std::chrono::steady_clock::now();
		multiply_omp(matrix1, matrix2, result, size, 2);
		auto end2 = std::chrono::steady_clock::now();
		cout << ">> Using omp 2: " << endl;
		file << ">> Using omp 2: " << endl;
		cout << "\tMatrix's size is: " << size << "x" << size << std::endl;
		file << "\tMatrix's size is: " << size << "x" << size << std::endl;
		cout << "\tDef meth\'s time: " << std::chrono::duration<double>(end2 - start2).count() << std::endl;		
		file << "\tDef meth\'s time: " << std::chrono::duration<double>(end2 - start2).count() << std::endl;

		start2 = std::chrono::steady_clock::now();
		multiply_omp(matrix1, matrix2, result, size, 4);
		end2 = std::chrono::steady_clock::now();
		cout << ">> Using omp 4: " << endl;	
		file << ">> Using omp 4: " << endl;
		cout << "\tMatrix's size is: " << size << "x" << size << std::endl;	
		file << "\tMatrix's size is: " << size << "x" << size << std::endl;
		cout << "\tDef meth\'s time: " << std::chrono::duration<double>(end2 - start2).count() << std::endl;	
		file << "\tDef meth\'s time: " << std::chrono::duration<double>(end2 - start2).count() << std::endl;

		start2 = std::chrono::steady_clock::now();
		multiply_omp(matrix1, matrix2, result, size, 8);
		end2 = std::chrono::steady_clock::now();
		cout << ">> Using omp 8: " << endl;
		file << ">> Using omp 8: " << endl;
		cout << "\tMatrix's size is: " << size << "x" << size << std::endl;
		file << "\tMatrix's size is: " << size << "x" << size << std::endl;
		cout << "\tDef meth\'s time: " << std::chrono::duration<double>(end2 - start2).count() << std::endl;	
		file << "\tDef meth\'s time: " << std::chrono::duration<double>(end2 - start2).count() << std::endl;

		start2 = std::chrono::steady_clock::now();
		multiply_omp(matrix1, matrix2, result, size, 12);
		end2 = std::chrono::steady_clock::now();
		cout << ">> Using omp 12: " << endl;
		file << ">> Using omp 12: " << endl;
		cout << "\tMatrix's size is: " << size << "x" << size << std::endl;
		file << "\tMatrix's size is: " << size << "x" << size << std::endl;
		cout << "\tDef meth\'s time: " << std::chrono::duration<double>(end2 - start2).count() << std::endl;
		file << "\tDef meth\'s time: " << std::chrono::duration<double>(end2 - start2).count() << std::endl;


		cout << "_______________________________" << endl;
		file << "_______________________________" << endl;

		if (size == MAX_SIZE)
			write_result_to_file("Mul.txt", result, MAX_SIZE);
		s++;
	}
	file.close();



	return 0;
}
