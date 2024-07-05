


nvcc -g -G --std=c++11 -gencode=arch=compute_80,code=\"sm_80,compute_80\"  -x cu -c  ../test.cu -o CMakeFiles/test.dir/test.cu.o  && make
nvcc --std=c++11 -gencode=arch=compute_80,code=\"sm_80,compute_80\"  -x cu -c  -Xptxas -O3,-v  ../test.cu -o CMakeFiles/test.dir/test.cu.o  && make

-x cu -c -Xptxas -O3,-v 



/usr/local/cuda/bin/nvcc  --std=c++17 -gencode=arch=compute_86,code=\"sm_86,compute_86\"  -x cu -c -Xptxas -O3,-v  /work/matrix_cal/test.cu -o CMakeFiles/test.dir/test.cu.o  && make
/usr/local/cuda/bin/nvcc -g -G --std=c++17 -gencode=arch=compute_86,code=\"sm_86,compute_86\"  -x cu -c  /work/matrix_cal/test.cu -o CMakeFiles/test.dir/test.cu.o  && make 


set cuda api_failures stop