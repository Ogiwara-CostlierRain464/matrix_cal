docker run -it -v ./:/work  -d --gpus all --name ogi nvidia/cuda:12.5.0-devel-ubuntu20.04 bash


/usr/local/cuda/bin/nvcc  --std=c++17 -gencode=arch=compute_86,code=\"sm_86,compute_86\"  -x cu -c -Xptxas -O3,-v  /work/matrix_cal/test.cu -o CMakeFiles/test.dir/test.cu.o  && make
/usr/local/cuda/bin/nvcc -g -G --std=c++17 -gencode=arch=compute_86,code=\"sm_86,compute_86\"  -x cu -c  /work/matrix_cal/test.cu -o CMakeFiles/test.dir/test.cu.o  && make 


/usr/local/cuda/bin/nvcc  --std=c++17 -gencode=arch=compute_75,code=\"sm_75,compute_75\"  -x cu -c -Xptxas -O3,-v  /work/matrix_cal/main.cu -o CMakeFiles/main.dir/main.cu.o  && make
/usr/local/cuda/bin/nvcc -g -G --std=c++17 -gencode=arch=compute_75,code=\"sm_75,compute_75\"  -x cu -c  /work/matrix_cal/main.cu -o CMakeFiles/test.dir/main.cu.o  && make

/usr/local/cuda/bin/nvcc -g -G --std=c++17 -gencode=arch=compute_80,code=\"sm_80,compute_80\"  -x cu   /work/matrix_cal/main.cu  -o main -lgflags
/usr/local/cuda/bin/nvcc --std=c++17 -gencode=arch=compute_80,code=\"sm_80,compute_80\"  -x cu  -Xptxas -O3,-v  /work/matrix_cal/main.cu  -o main -lgflags



nsys profile --gpu-metrics-device 0 ./main

cuda-gdb main
cuda-gdb --args ./rcc-d -run_naive_tc -d_model=768 -batch_size=500000  -sparse_ratio=10  -iter_num=1000

./rcc -run_naive_tc -d_model=768 -batch_size=32  -sparse_ratio=5  -iter_num=1000
./ccc -run_row -d_model=768 -batch_size=32 -sparse_ratio=5 -L=1  -iter_num=1000

ncu -o a --section MemoryWorkloadAnalysis ./rcc -run_naive_tc -d_model=768 -batch_size=32  -sparse_ratio=5  -iter_num=1
ncu -o b --section MemoryWorkloadAnalysis ./ccc -run_row -d_model=768 -batch_size=32 -sparse_ratio=5 -L=1  -iter_num=1

ncu -i .nuc-rep

set cuda api_failures stop

stack

bt