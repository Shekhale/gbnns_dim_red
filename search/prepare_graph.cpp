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

#include "search_function.h"

using namespace std;


int main(int argc, char **argv) {

    string dataset_name;
    string file_lat_name;
    if (argc == 3) {
        dataset_name = argv[1];
        file_lat_name = argv[2];
    } else {
        cout << " Need to specify parameters" << endl;
        return 1;
    }

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


    string path_data = "/mnt/data/shekhale/data/" + dataset_name + "/" + dataset_name;
    string path_models = "/mnt/data/shekhale/models/nns_graphs/" + dataset_name + "/" + dataset_name;


    std::vector<float> db_low = loadXvecs<float>( path_data + "_base_" + file_lat_name + ".fvecs", d_low, n);

    vector< vector <uint32_t>> knn_low =  load_edges(path_models + "_knn_1k_" + file_lat_name + ".ivecs", n, "knn_low");


	vector< vector <uint32_t>> gd_knn_low(n);
	gd_knn_low = hnswlikeGD(knn_low, db_low.data(), 30, n, d_low, &l2, true, false);
	cout << "GD_knn " << FindGraphAverageDegree(gd_knn_low) << endl;

    string dir_gd_knn_low = path_models +  "_gd_knn_" + file_lat_name + ".ivecs";
    const char *edge_gd_knn_low_dir = dir_gd_knn_low.c_str();
    write_edges(edge_gd_knn_low_dir, gd_knn_low);

    return 0;

}
