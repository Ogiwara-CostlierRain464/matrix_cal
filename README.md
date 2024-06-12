


nvcc -g -G --std=c++11 -gencode=arch=compute_80,code=\"sm_80,compute_80\"  -x cu -c  ../test.cu -o CMakeFiles/test.dir/test.cu.o  && make
nvcc --std=c++11 -gencode=arch=compute_80,code=\"sm_80,compute_80\"  -x cu -c  -Xptxas -O3,-v  ../test.cu -o CMakeFiles/test.dir/test.cu.o  && make

-x cu -c -Xptxas -O3,-v 