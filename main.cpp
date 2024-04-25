#include <mpi.h>
#include <stdio.h>
#include <algorithm>
#include <ctime>
#include <math.h>
#define MASTER_RANK 0

#define MASTER_TAG 1
#define WORKER_TAG 2

#define MICRO 1000000

#define NOT_ENOUGH_PROCESSES_NUM_ERROR 1

MPI_Status status;

#define MAX_SIZE 1000

int check = 1;

struct statistical_para {
	double std;
	double mean;
	double confi_inter_left;
	double confi_inter_right;
};

statistical_para count_statistic(int matrix[MAX_SIZE][MAX_SIZE], int size) {
    statistical_para ret{};
    int sum = 0;
    double sum_sq = 0;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            sum += matrix[i][j];
        }
    }
    ret.mean = (double)sum / (size * size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            sum_sq += ((double)matrix[i][j] - ret.mean) * ((double)matrix[i][j] - ret.mean);
        }
    }
    ret.std = sqrt(sum_sq);

    ret.confi_inter_left = ret.mean - 1.96 * (ret.std / sqrt(size));
    ret.confi_inter_right = ret.mean + 1.96 * (ret.std / sqrt(size));

    return ret;
}

void write_matrix_to_file(int matrix[MAX_SIZE][MAX_SIZE], const char* filename, int size) {
    FILE* file = fopen(filename, "w");

    if (file == NULL) {
        printf("Cannot open file\n");
        return;
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            fprintf(file, "%d ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    printf("Matrix was written to file \"%s\".\n", filename);
}

int GenerateRandomNumber(){
	return std::rand() % 9 + 1;
}


void FillMatrix(int matrix[MAX_SIZE][MAX_SIZE], int rows, int cols){
	for(int i = 0; i < cols; i ++){
        for(int j = 0; j < rows; j ++){
          matrix[i][j] = GenerateRandomNumber();
        }
    }
}

void PrintMatrix(int matrix[MAX_SIZE][MAX_SIZE], int rows, int cols){
	printf("\n");
	for(int i = 0; i < rows; i ++){
		for(int j = 0; j < cols; j ++){
			printf("%d ", matrix[i][j]);
		}
		printf("\n");
	}
}

int a[MAX_SIZE][MAX_SIZE];
int b[MAX_SIZE][MAX_SIZE];
int c[MAX_SIZE][MAX_SIZE];

int main(int argc, char *argv[])
{
	int communicator_size;
	int process_rank;
	int process_id;
	int offset;
	int rows_num;
	int workers_num;
	int remainder;
	int whole_part;
	int message_tag;
	int i;
	int j;
	int k;
	const int M_SIZE = atoi(argv[1]);

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &communicator_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

	if(communicator_size < 2){
		MPI_Abort(MPI_COMM_WORLD, NOT_ENOUGH_PROCESSES_NUM_ERROR);
	}

	if(process_rank == MASTER_RANK){
		printf("Generating matrixes\n");

		printf("\nGenerating matrix A with size %dx%d", M_SIZE, M_SIZE);
		FillMatrix(a, M_SIZE, M_SIZE);

		printf("\nGenerating matrix B with size %dx%d", M_SIZE, M_SIZE);
		FillMatrix(b, M_SIZE, M_SIZE);

		printf("\nStarting multiplication ... \n");
		long long int start = clock();

		workers_num = communicator_size - 1;
		whole_part = M_SIZE / workers_num;
		remainder = M_SIZE % workers_num;
		offset = 0;

		message_tag = MASTER_TAG;
		for(process_id = 1; process_id <= workers_num; process_id ++ ){
			rows_num = process_id <= remainder ? whole_part + 1 : whole_part;
			MPI_Send(&offset, 1, MPI_INT, process_id, message_tag, MPI_COMM_WORLD);
			MPI_Send(&rows_num, 1, MPI_INT, process_id, message_tag, MPI_COMM_WORLD);
			MPI_Send(&a[offset][0], rows_num * M_SIZE, MPI_INT, process_id, message_tag, MPI_COMM_WORLD);
			MPI_Send(&b, M_SIZE * M_SIZE, MPI_INT, process_id, message_tag, MPI_COMM_WORLD);

			offset += rows_num;
		}

		message_tag = WORKER_TAG;
		for(process_id = 1; process_id <= workers_num; process_id ++){
			MPI_Recv(&offset, 1, MPI_INT, process_id, message_tag, MPI_COMM_WORLD, &status);
			MPI_Recv(&rows_num, 1, MPI_INT, process_id, message_tag, MPI_COMM_WORLD, &status);
			MPI_Recv(&c[offset][0], rows_num * M_SIZE, MPI_INT, process_id, message_tag, MPI_COMM_WORLD, &status);
		}
		printf("\nResult A*B\n");
		long long int end = clock();
		double diff = (double)((end - start) / (1.0 * CLOCKS_PER_SEC));

		printf("\n%dx%d - %f seconds\n", M_SIZE, M_SIZE, diff);

		if (check) {
			write_matrix_to_file(a, "matrixA.txt", M_SIZE);
			write_matrix_to_file(b, "matrixB.txt", M_SIZE);
			write_matrix_to_file(c, "matrix_result.txt", M_SIZE);

			statistical_para st;
			st = count_statistic(a, M_SIZE);

			printf("\nStatistical parameters for matrix A\n");
			printf("Mean of matrix: %.2f\n", st.mean);
			printf("Standard deviation of matrix: %.2f\n", st.std);
			printf("Confidence interval with 95%% confidence: (%.2f, %.2f)\n", st.confi_inter_left, st.confi_inter_right);

			st = count_statistic(b, M_SIZE);

			printf("\nStatistical parameters for matrix B\n");
			printf("Mean of matrix: %.2f\n", st.mean);
			printf("Standard deviation of matrix: %.2f\n", st.std);
			printf("Confidence interval with 95%% confidence: (%.2f, %.2f)\n", st.confi_inter_left, st.confi_inter_right);
			check = 0;
		}
	}

	if(process_rank != MASTER_RANK){
		message_tag = MASTER_TAG;
		MPI_Recv(&offset, 1, MPI_INT, MASTER_RANK, message_tag, MPI_COMM_WORLD, &status);
		MPI_Recv(&rows_num, 1, MPI_INT, MASTER_RANK, message_tag, MPI_COMM_WORLD, &status);
		MPI_Recv(&a, rows_num * M_SIZE, MPI_INT, MASTER_RANK, message_tag, MPI_COMM_WORLD, &status);
		MPI_Recv(&b, M_SIZE * M_SIZE, MPI_INT, MASTER_RANK, message_tag, MPI_COMM_WORLD, &status);

		for(k = 0; k < M_SIZE; k ++){
			for(i = 0; i < rows_num; i ++){
				c[i][k] = 0;
				for(j = 0; j < M_SIZE; j ++){
					c[i][k] += a[i][j] * b[j][k];
				}
			}
		}

		message_tag = WORKER_TAG;
		MPI_Send(&offset, 1, MPI_INT, MASTER_RANK, message_tag, MPI_COMM_WORLD);
		MPI_Send(&rows_num, 1, MPI_INT, MASTER_RANK, message_tag, MPI_COMM_WORLD);
		MPI_Send(&c, rows_num * M_SIZE, MPI_INT, MASTER_RANK, message_tag, MPI_COMM_WORLD);
	}

	MPI_Finalize();


	return 0;
}
