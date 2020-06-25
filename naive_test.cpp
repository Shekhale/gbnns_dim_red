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
    int d_low_c;
    if (argc == 3) {
        dataset_name = argv[1];
        d_low_c = atoi(argv[2]);
    } else {
        cout << " Need to specify parameters" << endl;
        return 1;
    }

    int knn_size = 25;
    int n_q_c = 10000;
    int d_c = 128;
    string hnsw_name = "";
    vector<int> efs;
    vector<int> efs_hnsw;

    if (dataset_name == "sift") {
        vector<int> efs_c{40, 60, 80, 100, 120, 140, 160};
        efs.insert(efs.end(), efs_c.begin(), efs_c.end());
        vector<int> efs_hnsw_c{10, 20, 30, 40, 60, 80, 100};
        efs_hnsw.insert(efs.end(), efs_hnsw_c.begin(), efs_hnsw_c.end());
        hnsw_name = "M18_ef2000_onelevel1";
    } else if (dataset_name == "gist") {
        vector<int> efs_c{200, 400, 600, 800, 1000};
        efs.insert(efs.end(), efs_c.begin(), efs_c.end());
        vector<int> efs_hnsw_c{200, 400, 600, 800, 1000};
        efs_hnsw.insert(efs.end(), efs_hnsw_c.begin(), efs_hnsw_c.end());
        n_q_c = 1000;
        d_c = 960;
        hnsw_name = "M18_ef1000_onelevel1";
    } else if (dataset_name == "glove") {
        vector<int> efs_c{300, 400, 600, 800, 1000};
        efs.insert(efs.end(), efs_c.begin(), efs_c.end());
        vector<int> efs_hnsw_c{200, 400, 600, 800, 1000};
        efs_hnsw.insert(efs.end(), efs_hnsw_c.begin(), efs_hnsw_c.end());
        d_c = 300;
        hnsw_name = "M20_ef2000";
    } else if (dataset_name == "deep") {
        vector<int> efs_c{40, 80, 120, 160, 200};
        efs.insert(efs.end(), efs_c.begin(), efs_c.end());
        vector<int> efs_hnsw_c{40, 80, 120, 160, 200};
        efs_hnsw.insert(efs.end(), efs_hnsw_c.begin(), efs_hnsw_c.end());
        d_c = 96;
        hnsw_name = "M16_ef500_onelevel1";
    } else {
        cout << " Need to specify parameters for dataset" << endl;
        return 1;
    }

    time_t start, end;
    const size_t n = 1000000;  // number of points in base set
    const size_t n_q = n_q_c;
    const size_t n_tr = 100;
    const size_t d = d_c;  // dimension of data
    const size_t d_low = d_low_c;  // dimension of latent data
    const size_t kl_size = 15; // KL graph size

    L2Metric l2 = L2Metric();
    Angular ang = Angular();

    cout << "d = " << d << ", kl_size = " << kl_size << ", knn_size = " << knn_size <<  endl;

    std::mt19937 random_gen;
    std::random_device device;
    random_gen.seed(device());

    string path_data = "/mnt/data/shekhale/data/" + dataset_name + "/" + dataset_name;
    string path_models = "/mnt/data/shekhale/models/nns_graphs/" + dataset_name;
    string net_style = "naive_triplet";

    string dir_d = path_data + "_base.fvecs";
    const char *database_dir = dir_d.c_str();  // path to data
    string dir_q = path_data + "_query.fvecs";
    const char *query_dir = dir_q.c_str();  // path to data
    string dir_t = path_data + "_groundtruth.ivecs";
    const char *truth_dir = dir_t.c_str();  // path to data

    string data_low_dir_s = path_data + "_base_" + net_style + ".fvecs";
    const char *database_low_dir = data_low_dir_s.c_str();
    string query_low_dir_s = path_data + "_query_" + net_style + ".fvecs";
    const char *query_low_dir = query_low_dir_s.c_str();


    string dir_knn = path_models + "/knn_1k.ivecs";
    const char *edge_knn_dir = dir_knn.c_str();
    string dir_knn_low = path_models + "/knn_lat_1k_" + net_style + ".ivecs";
    const char *edge_knn_low_dir = dir_knn_low.c_str();

    string dir_kl = path_models + "/kl_sqrt_style.ivecs";
    const char *edge_kl_dir = dir_kl.c_str();
    string dir_kl_low = path_models + "/kl_lat_sqrt_style.ivecs";
    const char *edge_kl_low_dir = dir_kl_low.c_str();

    string edge_hnsw_dir_s = path_models + "/hnsw_" +  hnsw_name + ".ivecs";
    const char *edge_hnsw_dir = edge_hnsw_dir_s.c_str();

    string output = "results/naive_triplet_" + dataset_name + ".txt";
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

    std::vector<float> db_low(n * d_low);
    {
        std::ifstream data_input(database_low_dir, std::ios::binary);
        readXvec<float>(data_input, db_low.data(), d_low, n);
    }

    std::vector<float> queries_low(n_q * d_low);
    {
        std::ifstream data_input(query_low_dir, std::ios::binary);
        readXvec<float>(data_input, queries_low.data(), d_low, n_q);
    }


//--------------------------------------------------------------------------------------------------------------------------------------------


    vector< vector <uint32_t>> knn(n);
    knn = load_edges(edge_knn_dir, knn);
    knn = CutKNNbyK(knn, db, knn_size, n, d, &l2);
    cout << "knn " << FindGraphAverageDegree(knn) << endl;

    vector< vector <uint32_t>> knn_low(n);
    knn_low = load_edges(edge_knn_low_dir, knn_low);
    knn_low = CutKNNbyK(knn_low, db_low, knn_size, n, d_low, &l2);
    cout << "knn_low " << FindGraphAverageDegree(knn_low) << endl;

    bool kl_exist = FileExist(dir_kl);
    if (kl_exist != true) {
		KLgraph kl_sqrt;
		kl_sqrt.BuildByNumberCustom(kl_size, db, n, d, pow(n, 0.5), random_gen, &l2);
        write_edges(edge_kl_dir, kl_sqrt.longmatrixNN);

		KLgraph kl_sqrt_low;
		kl_sqrt_low.BuildByNumberCustom(kl_size, db_low, n, d_low, pow(n, 0.5), random_gen, &l2);
        write_edges(edge_kl_low_dir, kl_sqrt_low.longmatrixNN);
    }

    vector< vector <uint32_t>> kl(n);
    kl = load_edges(edge_kl_dir, kl);
    cout << "kl " << FindGraphAverageDegree(kl) << endl;

    vector< vector <uint32_t>> kl_low(n);
    kl_low = load_edges(edge_kl_low_dir, kl_low);
    cout << "kl_low " << FindGraphAverageDegree(kl_low) << endl;

    vector< vector <uint32_t>> hnsw(n);
    hnsw = load_edges(edge_hnsw_dir, hnsw);
    cout << "hnsw " << FindGraphAverageDegree(hnsw) << endl;

    if (dataset_name == "glove" or d % 8 != 0) {
        get_real_tests(n, d, d, n_q, n_tr, efs, random_gen, knn, knn, db, queries, db, queries, truth, output_txt, &ang, "knn", false, false, 1);
        get_real_tests(n, d, d, n_q, n_tr, efs_hnsw, random_gen, hnsw, hnsw, db, queries, db, queries, truth, output_txt, &ang, "hnsw", false, false, 1);
        get_real_tests(n, d, d, n_q, n_tr, efs, random_gen, knn, kl, db, queries, db, queries, truth, output_txt, &ang, "knn_kl", true, true, 1);
        get_real_tests(n, d, d_low, n_q, n_tr, efs, random_gen, knn_low, kl_low, db, queries, db_low, queries_low, truth, output_txt, &ang, "knn_kl_low", true, true, 1);
    } else {
        get_real_tests(n, d, d, n_q, n_tr, efs, random_gen, knn, knn, db, queries, db, queries, truth, output_txt, &l2, "knn", false, false, 1);
        get_real_tests(n, d, d, n_q, n_tr, efs_hnsw, random_gen, hnsw, hnsw, db, queries, db, queries, truth, output_txt, &l2, "hnsw", false, false, 1);
        get_real_tests(n, d, d, n_q, n_tr, efs, random_gen, knn, kl, db, queries, db, queries, truth, output_txt, &l2, "knn_kl", true, true, 1);
        get_real_tests(n, d, d_low, n_q, n_tr, efs, random_gen, knn_low, kl_low, db, queries, db_low, queries_low, truth, output_txt, &l2, "knn_kl_low", true, true, 1);
    }

    return 0;

}
