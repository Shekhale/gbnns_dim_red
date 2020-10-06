## Reducing Dimensionality method for Nearest Neighbor Search

and code for reproducing experiments from ICML2020 [paper](https://proceedings.icml.cc/static/paper_files/icml/2020/1229-Paper.pdf)

#### What does it do?

![img](https://github.com/Shekhale/gbnns_dim_red/raw/master/scheme_of_the_method.png)

 We introduce a new graph-based NNS method which is based on mapping the dataset into a space of lower dimension while trying to preserve local geometry.
 Graph-based NNS is performed in a low-dimensional space and several best candidates are then evaluated in the original space.
 Searching in the low-dimensional space is faster since distances can be computed more efficiently.
 Also, while learning the mapping, we enhance the distribution of points with desired properties, which further improves graph-based search. 
You can combine this method with any graph-based structure like [HNSW](https://github.com/nmslib/hnswlib) or [NGT](https://github.com/yahoojapan/NGT).

#### Files description and run examples
First, you need to specify paths to prospective data location in the following files and `dim_red/data.py`

`train.py` file aggregate different techniques for dimensionality reducing from **dim_red** folder. To learn your favorite method you need to specify it with other learning parameters.
For example, 

```sh
$ python train.py --database sift --method triplet --dout 32 --epochs 40 --batch_size 512
```
 will learn transformation in low dimension, save network (as classic PyTorch net and as matrixes for the following search) and transformed dataset.

If you want to additionally monitor the **search** accuracy of representation, set `val_freq_search` parameter more then 0 and see `wrap/wrap_readme.md` for additional instruction.

When a transformation is learned you need to build your favorite graph for low-dimensional data.
A possible solution is to use `search/prepare_graph.cpp`, which will use all available CPU and needed to specify '--save_knn_1k' parameter to 1 in the previous command.

Next run 
```sh
$ g++ -Ofast -std=c++11 -fopenmp -march=native -fpic -w -ftree-vectorize final_test.cpp
$ ./a.out sift
```
The last will start the searching procedure which will finally print and save results.
 
For draw results, you can use `results/draw_results.ipynb` notebook.

##### Learn | search faster 
If you want to further accelerate the code for learning, you can install [Faiss](https://github.com/facebookresearch/faiss) with GPU support

You can make the search procedure faster if use:

 - thinner or shorter net for transformation

 - the better implemented of matrix multiplication for net application (not implemented) 
 
 - fast matrix multiplication for re-ranking best elements from a low-dimensional space in original (not implemented) 

##### Learn better
For most task `triplet` method quiet enough, but if you:

 - want lightly increase the quality of representation 

 - (!) planning or must use **fixed** (the same) graph in low dimension

we recommend using `angular` method


#### Some results

![img](https://github.com/Shekhale/gbnns_dim_red/raw/master/results/real_datasets_results.png)


#### Naive learning for ICML2020 paper
 `train_naive_triplet.py` - main file for reproducing results from ICLM paper.
 It learns transformation in low dimension, saves transformed dataset,
 and builds kNN graph for search.
 Next, you must execute `search/naive_test.cpp` with a corresponding setup.
 
