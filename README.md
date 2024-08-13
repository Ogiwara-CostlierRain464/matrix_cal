docker run -it -v ./:/work  -d --gpus all --name ogi nvidia/cuda:12.5.0-devel-ubuntu20.04 bash


/usr/local/cuda/bin/nvcc  --std=c++17 -gencode=arch=compute_86,code=\"sm_86,compute_86\"  -x cu -c -Xptxas -O3,-v  /work/matrix_cal/test.cu -o CMakeFiles/test.dir/test.cu.o  && make
/usr/local/cuda/bin/nvcc -g -G --std=c++17 -gencode=arch=compute_86,code=\"sm_86,compute_86\"  -x cu -c  /work/matrix_cal/test.cu -o CMakeFiles/test.dir/test.cu.o  && make 


/usr/local/cuda/bin/nvcc  --std=c++17 -gencode=arch=compute_75,code=\"sm_75,compute_75\"  -x cu -c -Xptxas -O3,-v  /work/matrix_cal/main.cu -o CMakeFiles/main.dir/main.cu.o  && make
/usr/local/cuda/bin/nvcc -g -G --std=c++17 -gencode=arch=compute_75,code=\"sm_75,compute_75\"  -x cu -c  /work/matrix_cal/main.cu -o CMakeFiles/test.dir/main.cu.o  && make

/usr/local/cuda/bin/nvcc -g -G --std=c++17 -gencode=arch=compute_75,code=\"sm_75,compute_75\"  -x cu   /work/matrix_cal/main.cu  -o main


cuda-gdb main

set cuda api_failures stop

stack

bt