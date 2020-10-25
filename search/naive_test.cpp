
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
    const size_t kl_size = atoi(paramsMap["kl_size"].c_str());

    cout << n << " " << n_q << " " << n_tr << " " << d << " " << d_low << endl;
//
//    int knn_size = 25;
//    int n_q_c = 10000;

    vector<int> efs = getVectorFromString(paramsMap["efs"]);
    vector<int> efs_hnsw_origin = getVectorFromString(paramsMap["efs_hnsw"]);
    string hnsw_name = paramsMap["hnsw_name"];
    string netStyle = "naive";


    string pathData = "/mnt/data/shekhale/data/" + datasetName + "/" + datasetName;
    string pathModels = "/mnt/data/shekhale/models/nns_graphs/" + datasetName;

//-----------------------   LOAD DATA   ----------------------------------------------------

    vector<float> db = loadXvecs<float>(pathData + "_base.fvecs", d, n);

    vector<float> queries = loadXvecs<float>(pathData + "_query.fvecs", d, n_q);

    vector<uint32_t> truth = loadXvecs<uint32_t>(pathData + "_groundtruth.ivecs", n_tr, n_q);

    vector<float> db_low = loadXvecs<float>(pathData + "_base_" + netStyle + ".fvecs", d_low, n);
    vector<float> queries_low = loadXvecs<float>(pathData + "_query" + netStyle + ".fvecs", d, n_q);


// -----------------------   LOAD and CREATE GRAPHS  ------------------------------

    vector< vector <uint32_t>> hnsw =  loadEdges(pathModels + "/hnsw_" +  hnsw_name + ".ivecs", n, "hnsw");
    vector< vector <uint32_t>> knn =  loadEdges(pathModels + "/" + datasetName + "knn.ivecs", n, "knn");
    vector< vector <uint32_t>> knn_low =  loadEdges(pathModels + "/" + datasetName + "knn_low.ivecs", n, "knn_low");


    string kl_dir = pathModels + "/" + datasetName + "_kl_sqrt_style.ivecs";
    string kl_dir_low = pathModels + "/" + datasetName +  "_kl_llow_sqrt_style.ivecs";


    string output_s = "/home/shekhale/results/nns_graphs/" + datasetName + "/naive_results_" + datasetName + ".txt";
    const char *output = output_s.c_str();
    remove(output);

    bool klExist = checkFileExistence(kl_dir);
    if (klExist != true) {
        KLgraph kl_sqrt;
		kl_sqrt.BuildByNumberCustom(kl_size, db, n, d, pow(n, 0.5), random_gen, &l2);
        writeEdges(kl_dir, kl_sqrt.longmatrixNN);

		KLgraph kl_sqrt_low;
		kl_sqrt_low.BuildByNumberCustom(kl_size, db_low, n, d_low, pow(n, 0.5), random_gen, &l2);
        writeEdges(kl_dir_low, kl_sqrt_low.longmatrixNN);
    }

    vector< vector <uint32_t>> kl =  loadEdges(kl_dir, n, "kl");
    vector< vector <uint32_t>> kl_low =  loadEdges(kl_dir_low, n, "kl_low");


// -----------------------   SEARCHING PROCEDURE  ------------------------------

    int numberExper = 5;
    int numberThreads = 1;

    performRealTests(n, d, d, n_q, n_tr, efs_hnsw_origin, random_gen, hnsw, hnsw, db, queries, db, queries,
                     truth, output, &l2, "hnsw", false, false, numberExper, numberThreads);
    performRealTests(n, d, d, n_q, n_tr, efs_hnsw_origin, random_gen, knn, knn, db, queries, db, queries,
                     truth, output, &l2, "knn", false, false, numberExper, numberThreads);
    performRealTests(n, d, d, n_q, n_tr, efs_hnsw_origin, random_gen, knn, kl, db, queries, db, queries,
                     truth, output, &l2, "knn_lk", true, true, numberExper, numberThreads);
    performRealTests(n, d, d_low, n_q, n_tr, efs_hnsw_origin, random_gen, knn_low, kl_low, db, queries, db_low,
                     queries_low, truth, output, &l2, "knn_lk_low", true, true, numberExper, numberThreads);

    return 0;

}
