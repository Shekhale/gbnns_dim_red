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

#include "support_classes.h"
//#include "support_func.h"
#include "visited_list_pool.h"


using namespace std;


struct triple_result {
    priority_queue<pair<float, int > > topk;
    int hops;
    int dist_calc;
    int degree;
};


void MakeStep(vector <uint32_t> &graph_level, const float *query, const float* db,
              priority_queue<pair<float, int > > &topResults,
              priority_queue<std::pair<float, int > > &candidateSet,
              Metric *metric, uint32_t d, int &query_dist_calc, bool &found, int &ef, int &k,
              VisitedList *vl) {


    vl_type *massVisited = vl->mass;
    vl_type currentV = vl->curV;
    for (int j = 0; j < graph_level.size(); ++j) {
        int neig_num = graph_level[j];
        if (massVisited[neig_num] != currentV) {
            massVisited[neig_num] = currentV;
            const float *neig_coord = db + neig_num * d;
            float dist = metric->Dist(query, neig_coord, d);
            query_dist_calc++;

            if (topResults.top().first > dist || topResults.size() < ef) {
                candidateSet.emplace(-dist, neig_num);
                found = true;
                topResults.emplace(dist, neig_num);
                if (topResults.size() > ef)
                    topResults.pop();
            }
        }
    }
}


triple_result search(const float *query, const float* db, uint32_t N, uint32_t d,
                      vector<vector <uint32_t> > &main_graph, vector<vector <uint32_t> > &auxiliary_graph,
                      int ef, int k, vector<uint32_t> &inter_points, Metric *metric,
                     VisitedListPool *visitedlistpool,
                      bool use_second_graph, bool llf, uint32_t hops_bound) {


    std::priority_queue<std::pair<float, int > > topResults;

    int query_dist_calc = 1;
    int num_hops = 0;
    for (int i = 0; i < inter_points.size(); ++i) {
        std::priority_queue<std::pair<float, int > > candidateSet;
        const float* start = db + inter_points[i]*d;
        float dist = metric->Dist(query, start, d);

        topResults.emplace(dist, inter_points[i]);
        candidateSet.emplace(-dist, inter_points[i]);
        VisitedList *vl = visitedlistpool->getFreeVisitedList();
        vl_type *massVisited = vl->mass;
        vl_type currentV = vl->curV;
        massVisited[inter_points[i]] = currentV;
        while (!candidateSet.empty()) {
            std::pair<float, int> curr_el_pair = candidateSet.top();
            if (-curr_el_pair.first > topResults.top().first) break;

            candidateSet.pop();
            int curNodeNum = curr_el_pair.second;
            bool auxiliary_found = false;

            if (use_second_graph and num_hops < hops_bound) {
                vector <uint32_t> curAuxiliaryNodeNeighbors = auxiliary_graph[curNodeNum];
                MakeStep(curAuxiliaryNodeNeighbors, query, db,
                        topResults, candidateSet,
                        metric,
                        d, query_dist_calc, auxiliary_found, ef, k,
                        vl);
            }

            if (!(auxiliary_found * llf) or !use_second_graph) {
                vector <uint32_t> curMainNodeNeighbors = main_graph[curNodeNum];
                MakeStep(curMainNodeNeighbors, query, db,
                        topResults, candidateSet,
                        metric,
                        d, query_dist_calc, auxiliary_found, ef, k,
                        vl);
            }
            num_hops++;
        }
        visitedlistpool->releaseVisitedList(vl);
    }


    while (topResults.size() > k) {
        topResults.pop();
    }

    triple_result ans{topResults, num_hops, query_dist_calc};
    return ans;
}


int GetRealNearest(const float* point_q, int k, int d, int d_low, priority_queue<pair<float, int > > &topk,
                    vector<float> &ds,
                    Metric *metric) {

    const float* point_i = ds.data() + d * topk.top().second;
    float min_dist = metric->Dist(point_i, point_q, d);
    int real_topk = topk.top().second;
    topk.pop();
    float dist;
    while (!topk.empty()) {
        point_i = ds.data() + d * topk.top().second;
        dist = metric->Dist(point_i, point_q, d);
        if (dist < min_dist) {
            min_dist = dist;
            real_topk = topk.top().second;
        }
        topk.pop();
    }

    return real_topk;
}

void get_one_test(vector<vector<uint32_t> > &knn_graph, vector<vector<uint32_t> > &kl_graph,
                  vector<float> &ds, vector<float> &queries, vector<float> &ds_low, torch::jit::script::Module &net,
                  vector<uint32_t> &truth,
                  int n, int d, int d_low, int n_q, int n_tr, int ef, int k, string graph_name,
                  Metric *metric, const char* output_txt,
                  vector<vector<uint32_t> > inter_points, bool use_second_graph, bool llf, uint32_t hops_bound, int dist_calc_boost,
                  int recheck_size, int number_exper, int number_of_threads) {

    std::ofstream outfile;
    outfile.open(output_txt, std::ios_base::app);


    VisitedListPool *visitedlistpool = new VisitedListPool(1, n);
    int hops = 0;
    int dist_calc = 0 + dist_calc_boost * n_q;
    float acc = 0;
    float work_time = 0;
    int num_exp = 0;


//    omp_set_num_threads(number_of_threads);
    for (int v = 0; v < number_exper; ++v) {
        num_exp += 1;
        vector<int> ans(n_q);
        StopW stopw = StopW();
//#pragma omp parallel for
        for (int i = 0; i < n_q; ++i) {

            triple_result tr;
            const float* point_q = queries.data() + i * d;
//            const float* point_q_low = queries_low.data() + i * d_low;
            if (d != d_low) {
                std::vector<torch::jit::IValue> inputs;
                torch::Tensor query_tensor = torch::ones({1, d});
                for (int j=0; j < d; ++j) {
                    query_tensor[0][j] = queries[i*d + j];
                }
                inputs.push_back(query_tensor);
    //            const float* point_q_low = static_cast<float*>(net.forward(inputs).toTensor().data_ptr());
                at::Tensor output = net.forward(inputs).toTensor();
                float* point_q_low = static_cast<float*>(output.data_ptr());
				if (recheck_size > 0) {
	                tr = search(point_q_low, ds_low.data(), n, d_low, knn_graph, kl_graph, recheck_size,
	                            recheck_size, inter_points[i], metric, visitedlistpool, use_second_graph, llf, hops_bound);
	                ans[i] = GetRealNearest(point_q, k, d, d_low, tr.topk, ds, metric);
	                dist_calc += recheck_size;
				} else {
					tr = search(point_q_low, ds_low.data(), n, d_low, knn_graph, kl_graph, ef,
	                            k, inter_points[i], metric, visitedlistpool, use_second_graph, llf, hops_bound);

	                while (tr.topk.size() > k) {
	                    tr.topk.pop();
	                }
	                ans[i] = tr.topk.top().second;
				}
            } else {
                tr = search(point_q, ds.data(), n, d, knn_graph, kl_graph, ef,
                            k, inter_points[i], metric, visitedlistpool, use_second_graph, llf, hops_bound);

                while (tr.topk.size() > k) {
                    tr.topk.pop();
                }
                ans[i] = tr.topk.top().second;
            }

            hops += tr.hops;
            dist_calc += tr.dist_calc;
        }

        work_time += stopw.getElapsedTimeMicro();

        int print = 0;
        for (int i = 0; i < n_q; ++i) {
            acc += ans[i] == truth[i * n_tr];
        }
    }


    cout << "graph_type " << graph_name << " acc " << acc /  (num_exp * n_q) << " hops " << hops /  (num_exp * n_q) << " dist_calc "
         << dist_calc /  (num_exp * n_q) << " work_time " << work_time / (num_exp * 1e6 * n_q) << endl;
    outfile << "graph_type " << graph_name << " acc " << acc /  (num_exp * n_q) << " hops " << hops /  (num_exp * n_q) << " dist_calc "
            << dist_calc /  (num_exp * n_q) << " work_time " << work_time / (num_exp * 1e6 * n_q) << endl;
}




void get_real_tests(int n, int d, int d_low, int n_q, int n_tr, vector<int> efs, std::mt19937 random_gen,
                vector< vector<uint32_t> > &main_graph, vector< vector<uint32_t> > &kl, vector<float> &db,
                vector<float> &queries, vector<float> &db_low, torch::jit::script::Module &net, vector<uint32_t> &truth,
                const char* output_txt, Metric *metric,
                string graph_name, bool use_second_graph, bool llf, int number_exper) {

    vector<vector<uint32_t> > inter_points(n_q);
    int inter_points_mult = 1;
    if (graph_name.substr(0, 4) == "hnsw") {
        inter_points_mult = 0; // HNSW starts from 0
    }
    int num = 0;
    uniform_int_distribution<int> uniform_distr(0, n-1);
    for (int j=0; j < n_q; ++j) {
        num = uniform_distr(random_gen);
        inter_points[j].push_back(num * inter_points_mult);
    }

    uint32_t hops_bound = 11;

    int n_q_n = n_q;
    for (int i=0; i < efs.size(); ++i) {
        get_one_test(main_graph, kl, db, queries, db_low, net, truth, n, d, d_low, n_q_n, n_tr, efs[i], 1,
                     graph_name, metric, output_txt, inter_points, use_second_graph, llf, hops_bound, 0, efs[i], number_exper, 1);
    }

}

int main(int argc, char **argv) {

    torch::NoGradGuard no_grad;
//    char dataset = 's';
//    char dataset = 'd';
//    char dataset = 'g';
    char dataset = 'w';
    if (argc == 2) {
//        d_v = atoi(argv[1]);
        dataset = *argv[1];
    } else {
        cout << " Need to specify parameters" << endl;
        return 1;
    }

//    string transformation_type = string("triplet");
    string transformation_type = string("triplet");

    int n_q_p = 10000;
    int d_p = 128;
    int d_low_p = 32;
    string second_part = string("");
    string hnsw_name = string("");
    string hnsw_low_name = string("");
    int gd_graph_size = 50;
    vector<int> efs;
    string dataset_name = string("");

    if (dataset == 's') {
        dataset_name = string("sift");
//        vector<int> efs_n{40, 80, 120, 160};
        vector<int> efs_n{20, 40, 60, 80, 100, 120, 140, 160};
        efs.insert(efs.end(), efs_n.begin(), efs_n.end());
        d_p = 128;
        d_low_p = 32;
        second_part = string("_32_l_2_1m_5_10_w_4096_e_40");
        hnsw_name = string("M18_ef2000_onelevel1");
        hnsw_low_name = string("M18_ef500_onelevel1_low");
    } else if (dataset == 'g') {
        dataset_name = string("gist");
//        vector<int> efs_n{300, 500};
        vector<int> efs_n{200, 400, 600, 800, 1000};
        efs.insert(efs.end(), efs_n.begin(), efs_n.end());
        n_q_p = 1000;
        d_p = 960;
        d_low_p = 64;
//        second_part = string("_64_l_2_09m_5_40_w_1024_e_40");
        second_part = string("_64_l_2_1m_5_40_w_1024_e_40");
        hnsw_name = string("M18_ef1000_onelevel1");
        hnsw_low_name = string("M18_ef500_onelevel1_low");
    } else if (dataset == 'w') {
        dataset_name = string("glove");
//        vector<int> efs_n{300};
        vector<int> efs_n{300, 400, 600, 800, 1000};
//        vector<int> efs_n{300,  600, 1000};
        efs.insert(efs.end(), efs_n.begin(), efs_n.end());
        second_part = string("_144_l_1_1m_5_40_w_256_e_40");
        d_low_p = 144;
        d_p = 300;
//        hnsw_name = string("M18_ef2000_onelevel1");
        hnsw_name = string("M20_ef2000");
        hnsw_low_name = string("M20_ef500_onelevel1_low");
//        efs.push_back(1000);
//        gd_graph_size = 100;
    } else if (dataset == 'd') {
        dataset_name = string("deep");
//        vector<int> efs_n{20, 80, 120, 180};
        vector<int> efs_n{20, 40, 60, 80, 100, 120, 140, 160, 180, 200};
        efs.insert(efs.end(), efs_n.begin(), efs_n.end());
        second_part = string("_48_l_2_1m_5_10_w_4096_e_40");
        d_p = 96;
        d_low_p = 48;
//        hnsw_name = string("M18_ef2000_onelevel1");
        hnsw_name = string("M16_ef500_onelevel1");
        hnsw_low_name = string("M16_ef500_onelevel1_low");
    }



    time_t start, end;

    L2Metric l2 = L2Metric();
//    LikeL2Metric l2 = LikeL2Metric();
    Angular ang = Angular();

    std::mt19937 random_gen;
    std::random_device device;
    random_gen.seed(device());


    const size_t n = 1000000;  // number of points in base set
    const size_t n_q = n_q_p;  // number of points in base set
    const size_t n_tr = 100;  // number of points in base set
    const size_t d = d_p;  // data dim
    const size_t d_low = d_low_p;

    cout << n << " " << n_q << " " << n_tr << " " << d << " " << d_low << endl;

    string path_data = "/mnt/data/shekhale/data/" + dataset_name + "/" + dataset_name;
    string path_models = "/mnt/data/shekhale/models/nns_graphs/" + dataset_name;
//    string net_style = "triplet";

    string file_lat_name = transformation_type + second_part;

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
    string dir_gd_knn_low = path_models + "/" + dataset_name + "_gd_knn_" + file_lat_name + ".ivecs";
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

	vector<int>  trash;


    vector< vector <uint32_t>> gd_knn_low(n);
    gd_knn_low = load_edges(edge_gd_knn_low_dir, gd_knn_low);
	cout << "gd_knn_low " << FindGraphAverageDegree(gd_knn_low) << endl;


    string net_path = "/mnt/data/shekhale/models/nns_graphs/glove/glove_net_144_l_1_1m_5_40_w_256_e_40_src.pth";

    torch::jit::script::Module net;
    try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
        net = torch::jit::load(net_path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    char* graph_name = "";

    graph_name = "GD_low";
    cout << graph_name << endl;
    get_real_tests(n, d, d_low, n_q, n_tr, efs, random_gen, gd_knn_low, gd_knn_low, db, queries, db_low, net,\
                   truth, output_txt, &ang, graph_name, false, false, 1);



    vector< vector <uint32_t>> hnsw(n);
    hnsw = load_edges(edge_hnsw_dir, hnsw);
	cout << "hnsw_low " << FindGraphAverageDegree(hnsw) << endl;

    graph_name = "hnsw";
    cout << graph_name << endl;
    get_real_tests(n, d, d_low, n_q, n_tr, efs, random_gen, hnsw, hnsw, db, queries, db_low, net,\
                   truth, output_txt, &ang, graph_name, false, false, 1);


//    string net_path = "/mnt/data/shekhale/models/nns_graphs/glove/glove_net_144_l_1_1m_5_40_w_256_e_40_src.pth";
//
//    torch::jit::script::Module module;
//    try {
//    // Deserialize the ScriptModule from a file using torch::jit::load().
//        module = torch::jit::load(net_path);
//    }
//    catch (const c10::Error& e) {
//        std::cerr << "error loading the model\n";
//        return -1;
//    }
//    for (int i = 0; i < 3; ++i) {
//        float tmp = 0;
//        for (int j = 0; j < d; ++j) {
//            tmp += queries[i*d + j] * queries[i*d + j];
//        }
//        cout << tmp << endl;
//    }
//
//    for (int i = 0; i < 10; ++i) {
//        std::vector<torch::jit::IValue> inputs;
//        torch::Tensor query_tensor = torch::ones({1, d});
//        for (int j=0; j < d; ++j) {
//            query_tensor[0][j] = queries[i*d + j];
//        }
//        inputs.push_back(query_tensor);
//        at::Tensor output = module.forward(inputs).toTensor();
//        float* data = static_cast<float*>(output.data_ptr());
//
//        cout << data[0] << " " << data[1] << " " << data[2]  << " " << data[3]  << endl;
//        cout << output[0][0] << " " << output[0][1] << output[0][2] << " " << output[0][3]  << endl;
//        cout << queries_low[i * d_low + 0] << " " << queries_low[i * d_low + 1] << " " << queries_low[i * d_low + 2]  << " " << queries_low[i * d_low + 3]  << endl;
//        cout << queries[i * d + 0] << " " << queries[i * d + 1]  << endl;
//        cout << query_tensor[0][0] << " " << query_tensor[0][1]  << endl;
//        cout << endl;
//    }

    return 0;

}