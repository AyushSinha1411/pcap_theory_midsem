$ make
nvcc -O3 -o spmv_csr spmv_csr.cu
$ ./spmv_csr
Sparse Matrix (4x4):
5 0 0 1
0 8 0 0
0 0 3 0
4 0 0 9

Vector X: [1.0, 2.0, 3.0, 4.0]

Result Vector Y = A * X: [9.0, 16.0, 9.0, 40.0]
CPU Result for verification: [9.0, 16.0, 9.0, 40.0]
$ make transform_matrix
nvcc -O3 -o transform_matrix transform_matrix.cu
$ ./transform_matrix
Enter the number of rows: 4
Enter the number of columns: 3
Enter the matrix elements row by row:
2 3 4
5 1 2
3 3 2
2 1 4

Original Matrix:
2.00    3.00    4.00
5.00    1.00    2.00
3.00    3.00    2.00
2.00    1.00    4.00

Transformed Matrix:
Row 1 (power 1):
2.00    3.00    4.00
Row 2 (power 2):
25.00   1.00    4.00
Row 3 (power 3):
27.00   27.00   8.00
Row 4 (power 4):
16.00   1.00    256.00
$ make complement_matrix
nvcc -O3 -o complement_matrix complement_matrix.cu
$ ./complement_matrix
Enter the number of rows: 4
Enter the number of columns: 4
Enter the matrix elements row by row:
1 2 3 4
6 5 8 3
2 4 10 1
9 1 2 5

Input Matrix A:
1       2       3       4
6       5       8       3
2       4       10      1
9       1       2       5

Output Matrix B (non-border elements replaced with 1's complement):
1       2       3       4
6       2       7       3
2       3       5       1
9       1       2       5

Explanation:
5 -> 2 (1's complement)
8 -> 7 (1's complement)
4 -> 3 (1's complement)
10 -> 5 (1's complement)
$ bash
Student@dsl-13:~/Desktop/220905108/lab9$ make complement_matrix
nvcc -O3 -o complement_matrix complement_matrix.cu
Student@dsl-13:~/Desktop/220905108/lab9$ ./complement_matrix
Enter the number of rows: 4
Enter the number of columns: 4
Enter the matrix elements row by row:
1 2 3 4
6 5 8 4
2 4 10 1
9 1 2 5

Input Matrix A:
1       2       3       4
6       5       8       4
2       4       10      1
9       1       2       5

Output Matrix B (non-border elements replaced with 1's complement in binary):
1       2       3       4
6       10      111     4
2       11      101     1
9       1       2       5

Explanation:
5 -> 2 (decimal) -> 10 (binary form of 1's complement)
8 -> 7 (decimal) -> 111 (binary form of 1's complement)
4 -> 3 (decimal) -> 11 (binary form of 1's complement)
10 -> 5 (decimal) -> 101 (binary form of 1's complement)
Student@dsl-13:~/Desktop/220905108/lab9$ ./complement_matrix
Enter the number of rows: 4
Enter the number of columns: 4
Enter the matrix elements row by row:
1 2 3 4
6 5 8 3
2 4 10 1
9 1 2 5

Input Matrix A:
1       2       3       4
6       5       8       3
2       4       10      1
9       1       2       5

Output Matrix B (non-border elements replaced with 1's complement in binary):
1       2       3       4
6       10      111     3
2       11      101     1
9       1       2       5

Explanation:
5 -> 2 (decimal) -> 10 (binary form of 1's complement)
8 -> 7 (decimal) -> 111 (binary form of 1's complement)
4 -> 3 (decimal) -> 11 (binary form of 1's complement)
10 -> 5 (decimal) -> 101 (binary form of 1's complement)
Student@dsl-13:~/Desktop/220905108/lab9$ 