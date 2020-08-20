
Sometimes during experiments might be useful connect some functions from different languages.
In this folder you can find example of such connection.

Instead of running three consecutive programs you can execute 'train.py' with method 'triplet_wrap',
then file 'triplet_wrap.py'  will be called
and after network learning program will try to do search in learned space and write results in corresponding file.

To do so you must first execute following commands:
user_name:~/dim-red/wrap$ swig -python c_support.i
user_name:~/dim-red/wrap$ g++ -c -Ofast -std=c++11 -fopenmp -march=native -fpic -w -ftree-vectorize c_support_wrap.c c_support.cpp -I"python_path"
user_name:~/dim-red/wrap$ g++ -shared c_support.o c_support_wrap.o -o _c_support.so

(You must replace 'python_path' in second command with correct path. For example, '/home/shekhale/anaconda3/include/python3.7m' in my case)

For more details about python / c++ wrapping see swig tutorial http://www.swig.org/tutorial.html

