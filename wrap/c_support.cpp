
#include <random>
#include <iostream>
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

#include "../search/support_classes.h"
#include "../search/visited_list_pool.h"

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
                  vector<float> &ds, vector<float> &queries, vector<float> &ds_low, vector<float> &queries_low,
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

    omp_set_num_threads(number_of_threads);
    for (int v = 0; v < number_exper; ++v) {
        num_exp += 1;
        vector<uint32_t> ans(n_q);
        StopW stopw = StopW();
#pragma omp parallel for
        for (int i = 0; i < n_q; ++i) {

            triple_result tr;
            const float* point_q = queries.data() + i * d;
            const float* point_q_low = queries_low.data() + i * d_low;
            if (d != d_low) {
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
                vector<float> &queries, vector<float> &db_low, vector<float> &queries_low, vector<uint32_t> &truth,
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
    int number_of_threads = omp_get_max_threads() - 1;
    for (int i=0; i < efs.size(); ++i) {
        get_one_test(main_graph, kl, db, queries, db_low, queries_low, truth, n, d, d_low, n_q, n_tr, efs[i], 1,
                     graph_name, metric, output_txt, inter_points, use_second_graph, llf, hops_bound, 0, efs[i], number_exper, number_of_threads);
    }

}


int get_graphs_and_search_tests(char transform_type, char dataset, int d_p, int d_low_p, int n_q_p, char val, int n_val,
                                bool reverse_gd) {

    string file_name = string("");
    if (transform_type == 't') {
        file_name = string("triplet_wrap");
    } else if (transform_type == 'p') {
        file_name = string("pca");
    }

    int gd_graph_size = 50;
    vector<int> efs(0);
    string dataset_name = string("");
    if (dataset == 's') {
        dataset_name = string("sift");
        efs.push_back(150);
    } else if (dataset == 'g') {
        dataset_name = string("gist");
        efs.push_back(250);
        efs.push_back(500);
    } else if (dataset == 'w') {
        dataset_name = string("glove");
        efs.push_back(500);
//        efs.push_back(1000);
        gd_graph_size = 50;
    } else if (dataset == 'd') {
        dataset_name = string("deep");
        efs.push_back(100);
        efs.push_back(150);
    }

    string valid = string("");
    if (val == 'v') {
        valid = "_valid";
    }

    time_t start, end;

    L2Metric l2 = L2Metric();

    std::mt19937 random_gen;
    std::random_device device;
    random_gen.seed(device());


    int nn_const = 1000000;
    if (val == 'v') {
        nn_const = n_val;
    }
    const size_t n = nn_const;  // number of points in base set
//    const size_t n_q = 10000;  // number of points in base set
    const size_t n_q = n_q_p;  // number of points in base set
    const size_t n_tr = 100;  // number of points in base set
    const size_t d = d_p;  // data dim
//    const size_t d = 128;  // data dim
    const size_t d_low = d_low_p;

    cout << n << " " << n_q << " " << n_tr << " " << d << " " << d_low << endl;

    string path_data = "/mnt/data/shekhale/data/" + dataset_name + "/" + dataset_name;
    string path_models = "/mnt/data/shekhale/models/nns_graphs/" + dataset_name + "/";
//    string net_style = "naive_triplet";

    string dir_d = path_data + "_base" + valid + ".fvecs";
    const char *data_dir = dir_d.c_str();  // path to data
    string dir_q = path_data + "_query" + valid + ".fvecs";
    const char *query_dir = dir_q.c_str();  // path to data
    string dir_t = path_data + "_groundtruth" + valid + ".ivecs";
    const char *truth_dir = dir_t.c_str();  // path to data

    string dir_d_low = path_data + "_base_"  + file_name + valid + ".fvecs";
    const char *data_low_dir = dir_d_low.c_str();  // path to data
    string dir_q_low = path_data + "_query_"  + file_name + valid + ".fvecs";
    const char *query_low_dir = dir_q_low.c_str();  // path to data

    string edge_knn_low_dir_s = path_models + "knn_1k_" + file_name + valid + ".ivecs";
    const char *edge_knn_low_dir = edge_knn_low_dir_s.c_str();

    string output_txt_s = "/home/shekhale/results/dim_red/" + dataset_name + "/train_results_" +  file_name + ".txt";
    const char *output_txt = output_txt_s.c_str();
//    remove(output_txt);

//--------------------------------------------------------------------------------------------------------------------------------------------
//    //============
//    // Load Data
//    //============

    std::cout << "Loading data from " << data_dir << std::endl;
    std::vector<float> ds(n * d);
    {
        std::ifstream data_input(data_dir, std::ios::binary);
        readXvec<float>(data_input, ds.data(), d, n);
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

    std::vector<float> ds_low(n * d_low);
    if (d > d_low) {
        std::ifstream data_input(data_low_dir, std::ios::binary);
        readXvec<float>(data_input, ds_low.data(), d_low, n);
    } else {
    ds_low = ds;
    }

    std::vector<float> queries_low(n_q * d_low);
    if (d > d_low) {
        std::ifstream data_input(query_low_dir, std::ios::binary);
        readXvec<float>(data_input, queries_low.data(), d_low, n_q);
    } else {
        queries_low = queries;
    }


	vector<int>  trash;

    vector< vector <uint32_t>> knn_low(n);
    knn_low = load_edges(edge_knn_low_dir, knn_low);
	cout << "knn_low " << FindGraphAverageDegree(knn_low) << endl;

	vector< vector <uint32_t>> gd_knn_low(n);
    // gd_knn_low = load_edges(edge_gd_knn_low_dir, gd_knn_low);
	gd_knn_low = hnswlikeGD(knn_low, ds_low.data(), 20, n, d_low, &l2, reverse_gd);
	cout << "GD_knn_low " << FindGraphAverageDegree(gd_knn_low) << endl;
    // write_edges(edge_gd_knn_dir, gd_knn);

//	vector< vector <uint32_t>>  knn_low_50 = CutKNNbyK(knn_low, ds_low, 50, n, d_low, &l2);
//    cout << "knn_low " << FindGraphAverageDegree(knn_low_50) << endl;

//    cout << "knn " << endl;
//    get_real_tests(n, d, d_low, n_q, n_tr, efs, random_gen, knn_low_50, knn_low_50, ds, queries, ds_low, queries_low, truth, output_txt, &l2, "knn_50", false, false, 1);

    cout << "GD knn 20 " << endl;
    get_real_tests(n, d, d_low, n_q, n_tr, efs, random_gen, gd_knn_low, gd_knn_low, ds, queries, ds_low, queries_low, truth, output_txt, &l2, "gd_knn_20", false, false, 1);

    return 0;
}