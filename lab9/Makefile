NVCC = nvcc
NVCC_FLAGS = -O3

all: spmv_csr transform_matrix complement_matrix

spmv_csr: spmv_csr.cu
	$(NVCC) $(NVCC_FLAGS) -o spmv_csr spmv_csr.cu

transform_matrix: transform_matrix.cu
	$(NVCC) $(NVCC_FLAGS) -o transform_matrix transform_matrix.cu

complement_matrix: complement_matrix.cu
	$(NVCC) $(NVCC_FLAGS) -o complement_matrix complement_matrix.cu

clean:
	rm -f spmv_csr transform_matrix complement_matrix