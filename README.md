## New method for NNS based on Reducing Dimensionality

and code for reproducing experiments from ICML2020 [paper](https://proceedings.icml.cc/static/paper_files/icml/2020/1229-Paper.pdf)

#### What does it do?

![img](https://github.com/Shekhale/gbnns_dim_red/raw/master/scheme_of_the_method.png)

 We introduce a new graph-based NNS method which is based on mapping the dataset into a space of lower dimension while trying to preserve local geometry.
 Graph-based NNS is performed in a low-dimensional space and several best candidates are then evaluated in the original space.
 Searching in the low-dimensional space is faster since distances can be computed more efficiently.
 Also, while learning the mapping, we enhance the distribution of points with desired properties, which further improves graph-based search. 
We perform a thorough analysis of different heuristics that can be used to improve NNS algorithms and enrich our method with the most promising ones.
As a result, we get a new state-of-the-art NNS algorithm.

#### Files description and run examples
First, you need to specify paths to prospective data location in following files and `dim_red/data.py`

`train.py` file aggregate different techniques for dimensionality reducing from **dim_red** folder. To learn your favorite method you need to specify it with other learning parameters.
For example, 

```sh
$ python train.py --database sift --method triplet --dout 32 --epochs 40 --batch_size 512
```
 will be learns transformation in low dimension, save network and transformed dataset, create pytorch's script for search and finally will build `knn_1k_triplet.ivecs` graph as a base for more complicated ones.

As you can see from paper, method selecting is not trivial task. In our view, the best method to solve NNS problem for real-world (often huge) datasets located in `best_of_our_knowledge.py`. 

When transformation is learned you need to build the desired graphs via `search/prepare_graph.cpp`, that will be use all available CPU.

Next run 
```sh
$ cd search/build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ cmake --build . --config Release
$ make
$ OMP_NUM_THREADS=1 ./cpu_only_test w
```
The last will started search procedure which will finally print and save results.
 
For more details and instructions about loading a torchscript model in C++ see Pytorch [tutorial](https://pytorch.org/tutorials/advanced/cpp_export.html)

For draw results you can use `results/draw_results.ipynb` notebook.

##### Requirements 
If you want to further accelerate the code, you can install [Faiss](https://github.com/facebookresearch/faiss) with GPU support

for (other) needed libs see `requirements.txt`


#### Naive learning for ICML2020 paper
 `train_naive_triplet.py` - main file for reproducing results from ICLM paper. It learns transformation in low dimension, save transformed dataset and build knn graph for search.
 Next you must execute `naive_test.cpp` from **search** folder 
 
```sh
$ g++ -Ofast -std=c++11 -fopenmp -march=native -fpic -w -ftree-vectorize naive_test.cpp
$ ./a.out
```