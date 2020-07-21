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


#include <set>
#include <algorithm>
#include <ctime>

#include "search_function.h"

using  namespace std;

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

    string params_path = "/home/shekhale/dim-red/search/parameters_of_databases.txt";
    std::map<std::string, std::string> params_map = ReadSearchParams(params_path, dataset_name);

    const size_t n = atoi(params_map["n"].c_str());
    const size_t n_q = atoi(params_map["n_q"].c_str());
    const size_t n_tr = atoi(params_map["n_tr"].c_str());
    const size_t d = atoi(params_map["d"].c_str());
    const size_t d_low = atoi(params_map["d_low"].c_str());

    cout << n << " " << n_q << " " << n_tr << " " << d << " " << d_low << endl;

//    string second_part = "";
//    string hnsw_low_name = "";
//    int gd_graph_size = 50;
    vector<int> efs = VectorFromString(params_map["efs"]);
    vector<int> efs_hnsw = VectorFromString(params_map["efs_hnsw"]);
    string hnsw_name = params_map["hnsw_name"];

    string path_data = "/mnt/data/shekhale/data/" + dataset_name + "/" + dataset_name;
    string path_models = "/mnt/data/shekhale/models/nns_graphs/" + dataset_name + "/";
//    string net_style = "naive_triplet";

    string dir_d = path_data + "_base.fvecs";
    const char *database_dir = dir_d.c_str();  // path to data
    string dir_q = path_data + "_query.fvecs";
    const char *query_dir = dir_q.c_str();  // path to data
//    string dir_t = path_data + "_groundtruth_test.ivecs";
    string dir_t = path_data + "_groundtruth.ivecs";
    const char *truth_dir = dir_t.c_str();  // path to data

//    string data_low_dir_s = path_data + "_base_" + net_style + ".fvecs";
//    const char *database_low_dir = data_low_dir_s.c_str();
//    string query_low_dir_s = path_data + "_query_" + net_style + ".fvecs";
//    const char *query_low_dir = query_low_dir_s.c_str();


//    string dir_knn = path_models + dataset_name + "_knn_1k.ivecs";
//    const char *edge_knn_dir = dir_knn.c_str();
    string dir_knn_gd = path_models + dataset_name + "_gd_knn_50.ivecs";
    const char *edge_knn_gd_dir = dir_knn_gd.c_str();
//    string dir_knn_low = path_models + "/knn_lat_1k_" + net_style + ".ivecs";
//    const char *edge_knn_low_dir = dir_knn_low.c_str();

//    string dir_kl = path_models + "kl_sqrt_style.ivecs";
//    const char *edge_kl_dir = dir_kl.c_str();
//    string dir_kl_low = path_models + "kl_lat_sqrt_style.ivecs";
//    const char *edge_kl_low_dir = dir_kl_low.c_str();
//
    string edge_hnsw_dir_s = path_models + "hnsw_" +  hnsw_name + ".ivecs";
    const char *edge_hnsw_dir = edge_hnsw_dir_s.c_str();

    string output = "results/triplet_" + dataset_name + ".txt";
    const char *output_txt = output.c_str();

    remove(output_txt);



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
//
//    std::vector<float> db_low(n * d_low);
//    {
//        std::ifstream data_input(database_low_dir, std::ios::binary);
//        readXvec<float>(data_input, db_low.data(), d_low, n);
//    }



//--------------------------------------------------------------------------------------------------------------------------------------------


//    vector< vector <uint32_t>> knn(n);
//    knn = load_edges(edge_knn_dir, knn);
//    knn = CutKNNbyK(knn, db, knn_size, n, d, &l2);
//    cout << "knn " << FindGraphAverageDegree(knn) << endl;

    vector< vector <uint32_t>> knn_gd(n);
    knn_gd = load_edges(edge_knn_gd_dir, knn_gd);
    cout << "knn_gd " << FindGraphAverageDegree(knn_gd) << endl;
//
//    vector< vector <uint32_t>> knn_low(n);
//    knn_low = load_edges(edge_knn_low_dir, knn_low);
//    knn_low = CutKNNbyK(knn_low, db_low, knn_size, n, d_low, &l2);
//    cout << "knn_low " << FindGraphAverageDegree(knn_low) << endl;

//    bool kl_exist = FileExist(dir_kl);
//    if (kl_exist != true) {
//		KLgraph kl_sqrt;
//		kl_sqrt.BuildByNumberCustom(kl_size, db, n, d, pow(n, 0.5), random_gen, &l2);
//        write_edges(edge_kl_dir, kl_sqrt.longmatrixNN);
//
//		KLgraph kl_sqrt_low;
//		kl_sqrt_low.BuildByNumberCustom(kl_size, db_low, n, d_low, pow(n, 0.5), random_gen, &l2);
//        write_edges(edge_kl_low_dir, kl_sqrt_low.longmatrixNN);
//    }

//    vector< vector <uint32_t>> kl(n);
//    kl = load_edges(edge_kl_dir, kl);
//    cout << "kl " << FindGraphAverageDegree(kl) << endl;
//
//    vector< vector <uint32_t>> kl_low(n);
//    kl_low = load_edges(edge_kl_low_dir, kl_low);
//    cout << "kl_low " << FindGraphAverageDegree(kl_low) << endl;

    vector< vector <uint32_t>> hnsw(n);
    hnsw = load_edges(edge_hnsw_dir, hnsw);
    cout << "hnsw " << FindGraphAverageDegree(hnsw) << endl;

    if (dataset_name == "glove" or d % 8 != 0) {
        get_real_tests(n, d, d, n_q, n_tr, efs, random_gen, knn_gd, knn_gd, db, queries, db, queries, truth, output_txt, &ang, "knn_gd", false, false, 1);
        get_real_tests(n, d, d, n_q, n_tr, efs_hnsw, random_gen, hnsw, hnsw, db, queries, db, queries, truth, output_txt, &ang, "hnsw", false, false, 1);
    } else {
        cout << "H" << endl;
        get_real_tests(n, d, d, n_q, n_tr, efs, random_gen, knn_gd, knn_gd, db, queries, db, queries, truth, output_txt, &l2, "knn_gd", false, false, 1);
        get_real_tests(n, d, d, n_q, n_tr, efs_hnsw, random_gen, hnsw, hnsw, db, queries, db, queries, truth, output_txt, &l2, "hnsw", false, false, 1);
    }
    return 0;

}
