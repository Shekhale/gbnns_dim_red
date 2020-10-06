
If you want to additionally monitor the **search** accuracy of representation during learning:

 - first you must execute following commands:
```sh
user_name:~/gbnns_dim_red/wrap$ swig -python c_support.i
user_name:~/gbnns_dim_red/wrap$ g++ -c -Ofast -std=c++11 -fopenmp -march=native -fpic -w -ftree-vectorize c_support_wrap.c c_support.cpp -I"python_path"
user_name:~/gbnns_dim_red/wrap$ g++ -shared c_support.o c_support_wrap.o -o _c_support.so
```

(replace 'python_path' in second command with correct path. For example, '/home/username/anaconda3/include/python3.8')

 - when wrap files are ready, set `val_freq_search` parameter more than 0 and start learning.


For more details about python / c++ wrapping see swig tutorial http://www.swig.org/tutorial.html