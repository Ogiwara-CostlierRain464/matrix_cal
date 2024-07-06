docker run -it -v ./:/work  -d --gpus all --name ogi nvidia/cuda:12.5.0-devel-ubuntu20.04 bash


/usr/local/cuda/bin/nvcc  --std=c++17 -gencode=arch=compute_86,code=\"sm_86,compute_86\"  -x cu -c -Xptxas -O3,-v  /work/matrix_cal/test.cu -o CMakeFiles/test.dir/test.cu.o  && make
/usr/local/cuda/bin/nvcc -g -G --std=c++17 -gencode=arch=compute_86,code=\"sm_86,compute_86\"  -x cu -c  /work/matrix_cal/test.cu -o CMakeFiles/test.dir/test.cu.o  && make 


set cuda api_failures stop