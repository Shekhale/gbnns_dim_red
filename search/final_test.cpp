
#include "search_function.h"

using namespace std;


int main(int argc, char **argv) {

    string datasetName;
    if (argc == 2) {
        datasetName = argv[1];
    } else {
        cout << " Need to specify parameters" << endl;
        return 1;
    }

    cout << datasetName << endl;
    time_t start, end;

    L2Metric l2 = L2Metric();

    std::mt19937 random_gen;
    std::random_device device;
    random_gen.seed(device());

    string paramsPath = "/home/shekhale/gbnns_dim_red/search/parameters_of_databases.txt";
    std::map<string, string> paramsMap = readSearchParams(paramsPath, datasetName);

    const size_t n = atoi(paramsMap["n"].c_str());
    const size_t n_q = atoi(paramsMap["n_q"].c_str());
    const size_t n_tr = atoi(paramsMap["n_tr"].c_str());
    const size_t d = atoi(paramsMap["d"].c_str());
    const size_t d_low = atoi(paramsMap["d_low"].c_str());
    const size_t d_hidden = atoi(paramsMap["d_hidden"].c_str());

    cout << n << " " << n_q << " " << n_tr << " " << d << " " << d_low << endl;

    vector<int> efs = getVectorFromString(paramsMap["efs"]);
    vector<int> efs_hnsw_origin = getVectorFromString(paramsMap["efs_hnsw"]);
    string hnsw_name = paramsMap["hnsw_name"];



    string pathData = "/mnt/data/shekhale/data/" + datasetName + "/" + datasetName;
    string pathModels = "/mnt/data/shekhale/models/nns_graphs/" + datasetName;


//-----------------------   LOAD DATA   ----------------------------------------------------

    vector<float> db = loadXvecs<float>(pathData + "_base.fvecs", d, n);

    vector<float> queries = loadXvecs<float>(pathData + "_query.fvecs", d, n_q);

    vector<uint32_t> truth = loadXvecs<uint32_t>(pathData + "_groundtruth.ivecs", n_tr, n_q);

    vector<float> db_ar = loadXvecs<float>(pathData + "_base_angular_optimal.fvecs", d_low, n);


// -----------------------   LOAD GRAPHS  ------------------------------

    vector< vector <uint32_t>> hnsw =  loadEdges(pathModels + "/hnsw_" +  hnsw_name + ".ivecs", n, "hnsw");
    vector< vector <uint32_t>> hnsw_ar = loadEdges(pathModels + "/hnsw_" + hnsw_name + "_angular_optimal.ivecs",
                                                    n, "hnsw_ar");

// -----------------------   LOAD NETS  ------------------------------

    string pathNets = pathModels + "/" + datasetName + "_net_as_matrix";
    string pathARNets =  pathNets + "_angular_optimal";

    int numberExper = 5;
    int numberThreads = 1;

    Net net;
    net.layerFirst = loadXvecs<float>(pathARNets + "_1.fvecs", d + 1, d_hidden);
    net.layerSecond = loadXvecs<float>(pathARNets + "_2.fvecs", d_hidden + 1, d_hidden);
    net.layerFinal = loadXvecs<float>(pathARNets + "_3.fvecs", d_hidden + 1, d_low);

// -----------------------   SEARCHING PROCEDURE  ------------------------------

    string output_s = "/home/shekhale/results/nns_graphs/" + datasetName + "/final_results_" + datasetName + ".txt";
    const char *output = output_s.c_str();
    remove(output);

    performRealTests(n, d, d, n_q, n_tr, efs_hnsw_origin, random_gen, hnsw, hnsw, db, queries, db, queries,
                     truth, output, &l2, "hnsw", false, false, numberExper, numberThreads);

    performRealNetTests(n, d, d_low, n_q, n_tr, efs, random_gen, hnsw_ar, hnsw_ar, db, queries, db_ar, &net,
                           d_hidden, truth, output, &l2, "hnsw_new_ar", false, false, numberExper, numberThreads);

    return 0;

}
