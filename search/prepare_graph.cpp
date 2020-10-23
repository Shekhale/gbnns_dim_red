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

    string datasetName;
    string fileLatName;
    if (argc == 3) {
        datasetName = argv[1];
        fileLatName = argv[2];
    } else {
        cout << " Need to specify parameters" << endl;
        return 1;
    }

    L2Metric l2 = L2Metric();
    Angular ang = Angular();

    std::mt19937 random_gen;
    std::random_device device;
    random_gen.seed(device());

    string params_path = "/home/shekhale/gbnns_dim_red/search/parameters_of_databases.txt";
    std::map<string, string> params_map = readSearchParams(params_path, datasetName);

    const size_t n = atoi(params_map["n"].c_str());
    const size_t n_q = atoi(params_map["n_q"].c_str());
    const size_t n_tr = atoi(params_map["n_tr"].c_str());
    const size_t d = atoi(params_map["d"].c_str());
    const size_t d_low = atoi(params_map["d_low"].c_str());

    cout << n << " " << n_q << " " << n_tr << " " << d << " " << d_low << endl;


    string pathData = "/mnt/data/shekhale/data/" + datasetName + "/" + datasetName;
    string pathModels = "/mnt/data/shekhale/models/nns_graphs/" + datasetName + "/" + datasetName;


    std::vector<float> db_low = loadXvecs<float>(pathData + "_base_" + fileLatName + ".fvecs", d_low, n);

    vector< vector <uint32_t>> knn_low =  loadEdges(pathModels + "_knn_1k_" + fileLatName + ".ivecs", n, "knn_low");


	vector< vector <uint32_t>> gd_knn_low(n);
	gd_knn_low = hnswlikeGD(knn_low, db_low.data(), 30, n, d_low, &l2, true, false);
	cout << "GD_knn " << findGraphAverageDegree(gd_knn_low) << endl;


    writeEdges(pathModels + "_gd_knn_" + fileLatName + ".ivecs", gd_knn_low);

    return 0;

}
