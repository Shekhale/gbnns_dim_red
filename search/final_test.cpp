#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>
#include <ctime>
#include <queue>
#include <vector>
#include <omp.h>

#include <limits>
#include <sys/time.h>

#include <chrono>

#include <set>
#include <algorithm>

#include "search_function.h"

using namespace std;


int main(int argc, char **argv) {

    string dataset_name;
    if (argc == 2) {
        dataset_name = argv[1];
    } else {
        cout << " Need to specify parameters" << endl;
        return 1;
    }

    cout << dataset_name << endl;
    time_t start, end;

    L2Metric l2 = L2Metric();
    Angular ang = Angular();

    std::mt19937 random_gen;
    std::random_device device;
    random_gen.seed(device());

    string params_path = "/home/shekhale/gbnns_dim_red/search/parameters_of_databases.txt";
    std::map<std::string, std::string> params_map = ReadSearchParams(params_path, dataset_name);

    const size_t n = atoi(params_map["n"].c_str());
    const size_t n_q = atoi(params_map["n_q"].c_str());
    const size_t n_tr = atoi(params_map["n_tr"].c_str());
    const size_t d = atoi(params_map["d"].c_str());
    const size_t d_low = atoi(params_map["d_low"].c_str());
    const size_t d_hidden = atoi(params_map["d_hidden"].c_str());

    cout << n << " " << n_q << " " << n_tr << " " << d << " " << d_low << endl;

//    string second_part = "";
//    string hnsw_low_name = "";
//    int gd_graph_size = 50;
//    vector<int> efs{500, 700, 900};
//    vector<int> efs_hnsw_origin{500, 700, 900};
    vector<int> efs = VectorFromString(params_map["efs"]);
    vector<int> efs_hnsw_origin = VectorFromString(params_map["efs_hnsw"]);
//    string hnsw_name = params_map["hnsw_name"];
    string hnsw_name = params_map["hnsw_name"];
    string second_part = params_map["second_part"];



    string path_data = "/mnt/data/shekhale/data/" + dataset_name + "/" + dataset_name;
    string path_models = "/mnt/data/shekhale/models/nns_graphs/" + dataset_name;

//--------------------------------------------------------------------------------------------------------------------------------------------
//    //============
//    // Load Data
//    //============


//    std::cout << "Loading data from " << database_dir << std::endl;

    std::vector<float> db = loadXvecs<float>(path_data + "_base.fvecs", d, n);

    std::vector<float> queries = loadXvecs<float>(path_data + "_query.fvecs", d, n_q);

    std::vector<uint32_t> truth = loadXvecs<uint32_t>(path_data + "_groundtruth.ivecs", n_tr, n_q);

    std::vector<float> db_ar = loadXvecs<float>(path_data + "_base_angular_optimal.fvecs", d_low, n);

//    cout << " step " << endl;

// -----------------------   LOAD GRAPHS  ------------------------------

    vector< vector <uint32_t>> hnsw =  load_edges(path_models + "/hnsw_" +  hnsw_name + ".ivecs", n, "hnsw");
    vector< vector <uint32_t>> hnsw_ar =  load_edges(path_models + "/hnsw_" + hnsw_name + "_angular_optimal.ivecs", n, "hnsw_ar");



// -----------------------   LOAD NETS  ------------------------------

    string path_nets = path_models + "/" + dataset_name + "_net_as_matrix";
    string path_ar_nets =  path_nets + "_angular_optimal";
    std::vector<float> matrix_ar_1 = loadXvecs<float>(path_ar_nets + "_1.fvecs", d + 1, d_hidden);
    std::vector<float> matrix_ar_2 = loadXvecs<float>(path_ar_nets + "_2.fvecs", d_hidden + 1, d_hidden);
    std::vector<float> matrix_ar_3 = loadXvecs<float>(path_ar_nets + "_3.fvecs", d_hidden + 1, d_low);

    int number_exper = 5;
    int number_of_threads = 1;


// -----------------------   SEARCHING PROCEDURE  ------------------------------

    string output_txt_s = string("/home/shekhale/results/nns_graphs/") + dataset_name +
                          string("/final_results_") + dataset_name + string(".txt");
    const char *output_txt = output_txt_s.c_str();
    remove(output_txt);

    get_real_tests(n, d, d, n_q, n_tr, efs_hnsw_origin, random_gen, hnsw, hnsw, db, queries, db, queries,
                   truth, output_txt, &l2, "hnsw", false, false, number_exper, number_of_threads);

    get_real_tests_matrix(n, d, d_low, n_q, n_tr, efs, random_gen, hnsw_ar, hnsw_ar, db, queries, db_ar,
                          matrix_ar_1, matrix_ar_2, matrix_ar_3, d_hidden,
                          truth, output_txt, &l2, "hnsw_new_ar", false, false, number_exper, number_of_threads);

    return 0;

}
