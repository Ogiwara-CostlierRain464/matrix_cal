./build/rcc -run_naive_tc -d_model=768 -batch_size=32  -sparse_ratio=1 -iter_num=1000
./build/ccc -run_naive_cu -d_model=768 -batch_size=32  -sparse_ratio=1 -iter_num=1000
