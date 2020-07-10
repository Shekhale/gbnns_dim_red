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

#include <chrono>

#include <limits>
#include <sys/time.h>

#include <algorithm>
#include <ctime>

//#include "support_classes.h"
#include "search_function.h"


using namespace std;

void get_one_test_torch(vector<vector<uint32_t> > &knn_graph, vector<vector<uint32_t> > &kl_graph,
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


    for (int v = 0; v < number_exper; ++v) {
        num_exp += 1;
        vector<int> ans(n_q);
        StopW stopw = StopW();
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




void get_real_tests_torch(int n, int d, int d_low, int n_q, int n_tr, vector<int> efs, std::mt19937 random_gen,
                vector< vector<uint32_t> > &main_graph, vector< vector<uint32_t> > &kl, vector<float> &db,
                vector<float> &queries, vector<float> &db_low, torch::jit::script::Module &net, vector<uint32_t> &truth,
                const char* output_txt, Metric *metric,
                string graph_name, bool use_second_graph, bool llf, int number_exper) {


    torch::NoGradGuard no_grad;

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
        get_one_test_torch(main_graph, kl, db, queries, db_low, net, truth, n, d, d_low, n_q_n, n_tr, efs[i], 1,
                     graph_name, metric, output_txt, inter_points, use_second_graph, llf, hops_bound, 0, efs[i], number_exper, 1);
    }

}
