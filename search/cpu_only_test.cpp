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

#include <torch/script.h>
#include <chrono>

#include <set>
#include <algorithm>

#include "search_function_torch.h"

using namespace std;


int main(int argc, char **argv) {

    string dataset_name;
    if (argc == 2) {
        dataset_name = argv[1];
    } else {
        cout << " Need to specify parameters" << endl;
        return 1;
    }

    string transformation_type = string("triplet");
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

    cout << n << " " << n_q << " " << n_tr << " " << d << " " << d_low << endl;

//    string second_part = string("");
//    string hnsw_name = string("");
//    string hnsw_low_name = string("");
//    int gd_graph_size = 50;
    vector<int> efs = VectorFromString(params_map["efs"]);


    string path_data = "/mnt/data/shekhale/data/" + dataset_name + "/" + dataset_name;
    string path_models = "/mnt/data/shekhale/models/nns_graphs/" + dataset_name;

    string file_lat_name = transformation_type + second_part;
    string net_path = path_models + "/" + dataset_name + "_net" + second_part + "_scr.pth";

    string dir_d = path_data + "_base.fvecs";
    const char *database_dir = dir_d.c_str();  // path to data
    string dir_q = path_data + "_query.fvecs";
    const char *query_dir = dir_q.c_str();  // path to data
    string dir_t = path_data + "_groundtruth.ivecs";
    const char *truth_dir = dir_t.c_str();  // path to data

    string data_low_dir_s = path_data + "_base_" + file_lat_name + ".fvecs";
    const char *database_low_dir = data_low_dir_s.c_str();
    string query_low_dir_s = path_data + "_query_" + file_lat_name + ".fvecs";
    const char *query_low_dir = query_low_dir_s.c_str();




//    string dir_knn = path_models + "/knn_25.ivecs";
//    const char *edge_knn_dir = dir_knn.c_str();
    string dir_gd_knn_low = path_models + "/" + dataset_name + "_gd_knn_50_" + file_lat_name + ".ivecs";
    const char *edge_gd_knn_low_dir = dir_gd_knn_low.c_str();

    string edge_hnsw_dir_s = path_models + "/hnsw_" +  hnsw_name + ".ivecs";
    const char *edge_hnsw_dir = edge_hnsw_dir_s.c_str();

    string output_txt_s = string("/home/shekhale/results/nns_graphs/") + dataset_name + string("/some_results_") + dataset_name + string(".txt");
    const char *output_txt = output_txt_s.c_str();
    remove(output_txt);

//--------------------------------------------------------------------------------------------------------------------------------------------
//    //============
//    // Load Data
//    //============

    std::cout << "Loading data from " << database_dir << std::endl;
    std::vector<float> db(n * d);
    {
        std::ifstream data_input(database_dir, std::ios::binary);
        readXvec<float>(data_input, db.data(), d, n);
    }

    std::vector<float> queries(n_q * d);
    {
        std::ifstream data_input(query_dir, std::ios::binary);
        readXvec<float>(data_input, queries.data(), d, n_q);
    }

    std::vector<uint32_t> truth(n_q * n_tr);
    {
        std::ifstream data_input(truth_dir, std::ios::binary);
        readXvec<uint32_t>(data_input, truth.data(), n_tr, n_q);
    }

    std::vector<float> db_low(n * d_low);
    if (d > d_low) {
        std::ifstream data_input(database_low_dir, std::ios::binary);
        readXvec<float>(data_input, db_low.data(), d_low, n);
    } else {
    db_low = db;
    }

    std::vector<float> queries_low(n_q * d_low);
    if (d > d_low) {
        std::ifstream data_input(query_low_dir, std::ios::binary);
        readXvec<float>(data_input, queries_low.data(), d_low, n_q);
    } else {
        queries_low = queries;
    }

// UPLOAD Graphs and net

    vector< vector <uint32_t>> gd_knn_low(n);
    gd_knn_low = load_edges(edge_gd_knn_low_dir, gd_knn_low);
	cout << "gd_knn_low " << FindGraphAverageDegree(gd_knn_low) << endl;

    torch::jit::script::Module net;
    try {
        net = torch::jit::load(net_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

//    vector< vector <uint32_t>> hnsw(n);
//    hnsw = load_edges(edge_hnsw_dir, hnsw);
//	cout << "hnsw " << FindGraphAverageDegree(hnsw) << endl;

// Get search

    if (dataset_name == "glove" or d % 8 != 0) {
        get_real_tests_torch(n, d, d_low, n_q, n_tr, efs, random_gen, gd_knn_low, gd_knn_low, db, queries, db_low, net,\
                   truth, output_txt, &ang, "GD_low", false, false, 1);
//        get_real_tests_torch(n, d, d_low, n_q, n_tr, efs, random_gen, hnsw, hnsw, db, queries, db_low, net,\
//                   truth, output_txt, &ang, "hnsw", false, false, 1);
    } else {
        get_real_tests_torch(n, d, d_low, n_q, n_tr, efs, random_gen, gd_knn_low, gd_knn_low, db, queries, db_low, net,\
                   truth, output_txt, &l2, "GD_low", false, false, 1);
//        get_real_tests_torch(n, d, d_low, n_q, n_tr, efs, random_gen, hnsw, hnsw, db, queries, db_low, net,\
//                   truth, output_txt, &l2, "hnsw", false, false, 1);
    }

    return 0;

}