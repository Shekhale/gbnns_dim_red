#include "support_classes.h"
#include "visited_list_pool.h"

using namespace std;


struct TripleResult {
    priority_queue<pair<float, int > > topk;
    int hops;
    int dist_calc;
    int degree;
};


void makeStep(vector <uint32_t> &graph_level, const float *query, const float* db,
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


TripleResult getOneSearchResults(const float *query, const float* db, uint32_t N, uint32_t d,
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
                makeStep(curAuxiliaryNodeNeighbors, query, db,
                        topResults, candidateSet,
                        metric,
                        d, query_dist_calc, auxiliary_found, ef, k,
                        vl);
            }

            if (!(auxiliary_found * llf) or !use_second_graph) {
                vector <uint32_t> curMainNodeNeighbors = main_graph[curNodeNum];
                makeStep(curMainNodeNeighbors, query, db,
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

    TripleResult ans{topResults, num_hops, query_dist_calc};
    return ans;
}


int getRealNearest(const float* point_q, int k, int d, int d_low, priority_queue<pair<float, int > > &topk,
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


void performTest(vector<vector<uint32_t> > &knn_graph, vector<vector<uint32_t> > &kl_graph,
                  vector<float> &ds, vector<float> &queries, vector<float> &ds_low, vector<float> &queries_low,
                  vector<uint32_t> &truth,
                  int n, int d, int d_low, int n_q, int n_tr, int ef, int k, string graph_name,
                  Metric *metric, const char* output_txt,
                  vector<vector<uint32_t> > inter_points, bool use_second_graph, bool llf, uint32_t hops_bound,
                  int dist_calc_boost, int recheck_size, int number_exper, int number_of_threads) {

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

            TripleResult tripletResult;
            const float* point_q = queries.data() + i * d;
            const float* point_q_low = queries_low.data() + i * d_low;
            if (d != d_low) {
				if (recheck_size > 0) {
	                tripletResult = getOneSearchResults(point_q_low, ds_low.data(), n, d_low, knn_graph, kl_graph,
	                                    recheck_size, recheck_size, inter_points[i], metric, visitedlistpool,
	                                    use_second_graph, llf, hops_bound);
	                ans[i] = getRealNearest(point_q, k, d, d_low, tripletResult.topk, ds, metric);
	                dist_calc += recheck_size;
				} else {
					tripletResult = getOneSearchResults(point_q_low, ds_low.data(), n, d_low, knn_graph, kl_graph, ef,
	                                    k, inter_points[i], metric, visitedlistpool, use_second_graph, llf, hops_bound);

	                while (tripletResult.topk.size() > k) {
	                    tripletResult.topk.pop();
	                }
	                ans[i] = tripletResult.topk.top().second;
				}
            } else {
                tripletResult = getOneSearchResults(point_q, ds.data(), n, d, knn_graph, kl_graph, ef,
                            k, inter_points[i], metric, visitedlistpool, use_second_graph, llf, hops_bound);

                while (tripletResult.topk.size() > k) {
                    tripletResult.topk.pop();
                }
                ans[i] = tripletResult.topk.top().second;
            }

            hops += tripletResult.hops;
            dist_calc += tripletResult.dist_calc;
        }

        work_time += stopw.getElapsedTimeMicro();

        int print = 0;
        for (int i = 0; i < n_q; ++i) {
            acc += ans[i] == truth[i * n_tr];
            // part for SIFT dataset bug fixing
            {
                const float *point_tr_f = ds.data() + d * truth[i * n_tr];
                const float *point_tr_s = ds.data() + d * truth[i * n_tr + 1];
    //            point_tr_t = ds.data() + d * truth[i * n_tr + 2];
                float dist = metric->Dist(point_tr_f, point_tr_s, d);
                if (dist == 0 and truth[i * n_tr] != truth[i * n_tr + 1]) {
                    acc += ans[i] == truth[i * n_tr + 1];
                }
            }
        }
    }

    cout << "graph_type " << graph_name << " acc " << acc /  (num_exp * n_q) << " hops " << hops /  (num_exp * n_q)
        << " dist_calc " << dist_calc /  (num_exp * n_q) << " work_time " << work_time / (num_exp * 1e6 * n_q) << endl;
    outfile << "graph_type " << graph_name << " acc " << acc /  (num_exp * n_q) << " hops " << hops /  (num_exp * n_q)
        << " dist_calc " << dist_calc /  (num_exp * n_q) << " work_time " << work_time / (num_exp * 1e6 * n_q) << endl;
}



void performSyntheticTests(int n, int d, int n_q, int n_tr, std::mt19937 random_gen,
                vector< vector<uint32_t> > &knn, vector< vector<uint32_t> > &kl, vector<float> &db,
                vector<float> &queries, vector<uint32_t> &truth, const char* output_txt,
                Metric *metric, string graph_name, bool use_second_graph, bool llf, bool beam_search) {

    vector<vector<uint32_t> > inter_points(n_q);
    int num = 0;
//    uniform_int_distribution<int> uniform_distr(0, n-1);
//    for (int j=0; j < n_q; ++j) {
//        num = uniform_distr(random_gen);
//        inter_points[j].push_back(num);
//    }

    vector<int> ef_coeff;
    vector<int> k_coeff;
    uint32_t hops_bound = 11;
    int recheck_size = -1;
    int knn_size = findGraphAverageDegree(knn);

    if (beam_search) {
        vector<int> k_coeff_{knn_size, knn_size, knn_size, knn_size, knn_size, knn_size};
        k_coeff.insert(k_coeff.end(), k_coeff_.begin(), k_coeff_.end());
    } else {
        vector<int> ef_coeff_{1, 1, 1, 1, 1, 1};
        ef_coeff.insert(ef_coeff.end(), ef_coeff_.begin(), ef_coeff_.end());
    }

    if (d == 3) {
        if (beam_search) {
            vector<int> ef_coeff_{10, 15, 20, 25, 30};
            ef_coeff.insert(ef_coeff.end(), ef_coeff_.begin(), ef_coeff_.end());
        } else {
            vector<int> k_coeff_{12, 14, 16, 18, 20};
            k_coeff.insert(k_coeff.end(), k_coeff_.begin(), k_coeff_.end());
        }
        hops_bound = 11;
    } else if (d == 5) {
        if (beam_search) {
            vector<int> ef_coeff_{7, 10, 15, 22, 25, 30};
            ef_coeff.insert(ef_coeff.end(), ef_coeff_.begin(), ef_coeff_.end());
        } else {
            vector<int> k_coeff_{15, 20, 25, 30, 40, 60};
            k_coeff.insert(k_coeff.end(), k_coeff_.begin(), k_coeff_.end());
        }
        hops_bound = 7;
    } else if (d == 9) {
        if (beam_search) {
            vector<int> ef_coeff_{5, 8, 15, 25, 30, 35};
            ef_coeff.insert(ef_coeff.end(), ef_coeff_.begin(), ef_coeff_.end());
        } else {
            vector<int> k_coeff_{60, 100, 150, 200, 250, 300};
            k_coeff.insert(k_coeff.end(), k_coeff_.begin(), k_coeff_.end());
        }
        hops_bound = 5;
    } else if (d == 17) {
        if (beam_search) {
            vector<int> ef_coeff_{10, 40, 70, 100, 130, 160};
            ef_coeff.insert(ef_coeff.end(), ef_coeff_.begin(), ef_coeff_.end());
        } else {
            vector<int> k_coeff_{750, 1000, 1250, 1500, 1750, 2000};
            k_coeff.insert(k_coeff.end(), k_coeff_.begin(), k_coeff_.end());
        }
        hops_bound = 4;
    }

    int exp_size = min(ef_coeff.size(), k_coeff.size());

    for (int i=0; i < exp_size; ++i) {
        vector< vector <uint32_t>> knn_cur = cutKNNbyK(knn, db.data(), k_coeff[i], n, d, metric);
        performTest(knn_cur, kl, db, queries, db, queries, truth, n, d, d, n_q, n_tr, ef_coeff[i], 1,
                    graph_name, metric, output_txt, inter_points, use_second_graph, llf, hops_bound, 0, recheck_size,
                    1, omp_get_max_threads());
    }
}



void performRealTests(int n, int d, int d_low, int n_q, int n_tr, vector<int> efs, std::mt19937 random_gen,
                vector< vector<uint32_t> > &main_graph, vector< vector<uint32_t> > &kl, vector<float> &db,
                vector<float> &queries, vector<float> &db_low, vector<float> &queries_low, vector<uint32_t> &truth,
                const char* output_txt, Metric *metric,
                string graph_name, bool use_second_graph, bool llf, int number_exper, int number_of_threads) {

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

    uint32_t hops_bound = 50;

    for (int i=0; i < efs.size(); ++i) {
        performTest(main_graph, kl, db, queries, db_low, queries_low, truth, n, d, d_low, n_q, n_tr, efs[i], 1,
                     graph_name, metric, output_txt, inter_points, use_second_graph, llf, hops_bound, 0, efs[i], number_exper, number_of_threads);
    }

}


void performNetTest(vector<vector<uint32_t> > &knn_graph, vector<vector<uint32_t> > &kl_graph,
                  vector<float> &ds,  vector<float> &queries, vector<float> &ds_low, const Net* net, size_t d_hidden,
                  vector<uint32_t> &truth,
                  int n, int d, int d_low, int n_q, int n_tr, int ef, int k, string graph_name,
                  Metric *metric, const char* output_txt,
                  vector<vector<uint32_t> > inter_points, bool use_second_graph, bool llf, uint32_t hops_bound,
                  int dist_calc_boost, int recheck_size, int number_exper, int number_of_threads) {

    std::ofstream outfile;
    outfile.open(output_txt, std::ios_base::app);


    VisitedListPool *visitedlistpool = new VisitedListPool(1, n);
    int hops = 0;
    int dist_calc = 0 + dist_calc_boost * n_q;
    float acc = 0;
    float work_time = 0;
    int num_exp = 0;

//    int d_hidden = 256;
    Angular ang = Angular();
    L2Metric l2 = L2Metric();
    vector<float> zeros(d_low);
    omp_set_num_threads(number_of_threads);
    for (int v = 0; v < number_exper; ++v) {
        num_exp += 1;
        vector<uint32_t> ans(n_q);
        StopW stopw = StopW();
//#pragma omp parallel for
        for (int i = 0; i < n_q; ++i) {

            TripleResult tripletResult;
            const float* point_q = queries.data() + i * d;
//            const float* point_q_low = queries_low.data() + i * d_low;
            if (d != d_low) {
                vector<float> queries_low(d_low);
                GetLowQueryFromNet(net, point_q, queries_low, zeros.data(), d, d_hidden, d_hidden, d_low, &ang, &l2);
//                const float* point_q_low = queries_low.data();
				if (recheck_size > 0) {
	                tripletResult = getOneSearchResults(queries_low.data(), ds_low.data(), n, d_low, knn_graph,
	                                    kl_graph, recheck_size, recheck_size, inter_points[i], metric, visitedlistpool,
	                                    use_second_graph, llf, hops_bound);
	                ans[i] = getRealNearest(point_q, k, d, d_low, tripletResult.topk, ds, metric);
	                dist_calc += recheck_size;
				} else {
					tripletResult = getOneSearchResults(queries_low.data(), ds_low.data(), n, d_low, knn_graph,
					                    kl_graph, ef, k, inter_points[i], metric, visitedlistpool, use_second_graph,
					                    llf, hops_bound);

	                while (tripletResult.topk.size() > k) {
	                    tripletResult.topk.pop();
	                }
	                ans[i] = tripletResult.topk.top().second;
				}
            } else {
                tripletResult = getOneSearchResults(point_q, ds.data(), n, d, knn_graph, kl_graph, ef,
                                    k, inter_points[i], metric, visitedlistpool, use_second_graph, llf, hops_bound);

                while (tripletResult.topk.size() > k) {
                    tripletResult.topk.pop();
                }
                ans[i] = tripletResult.topk.top().second;
            }

            hops += tripletResult.hops;
            dist_calc += tripletResult.dist_calc;
        }

        work_time += stopw.getElapsedTimeMicro();

        int print = 0;
        for (int i = 0; i < n_q; ++i) {
            acc += ans[i] == truth[i * n_tr];
            // part for SIFT dataset bug fixing
            {
                const float *point_tr_f = ds.data() + d * truth[i * n_tr];
                const float *point_tr_s = ds.data() + d * truth[i * n_tr + 1];
                float dist = metric->Dist(point_tr_f, point_tr_s, d);
                if (dist == 0 and truth[i * n_tr] != truth[i * n_tr + 1]) {
                    acc += ans[i] == truth[i * n_tr + 1];
                }
            }
        }
    }

    cout << "graph_type " << graph_name << " acc " << acc /  (num_exp * n_q) << " hops " << hops /  (num_exp * n_q)
        << " dist_calc " << dist_calc /  (num_exp * n_q) << " work_time " << work_time / (num_exp * 1e6 * n_q) << endl;
    outfile << "graph_type " << graph_name << " acc " << acc /  (num_exp * n_q) << " hops " << hops /  (num_exp * n_q)
        << " dist_calc " << dist_calc /  (num_exp * n_q) << " work_time " << work_time / (num_exp * 1e6 * n_q) << endl;
}


void performRealNetTests(int n, int d, int d_low, int n_q, int n_tr, vector<int> efs, std::mt19937 random_gen,
                vector< vector<uint32_t> > &main_graph, vector< vector<uint32_t> > &kl, vector<float> &db,
                vector<float> &queries, vector<float> &db_low, const Net* net, size_t d_hidden, vector<uint32_t> &truth,
                const char* output_txt, Metric *metric,
                string graph_name, bool use_second_graph, bool llf, int number_exper, int number_of_threads) {

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

    uint32_t hops_bound = 50;

    for (int i=0; i < efs.size(); ++i) {
        performNetTest(main_graph, kl, db, queries, db_low, net, d_hidden, truth, n, d, d_low,
                          n_q, n_tr, efs[i], 1, graph_name, metric, output_txt, inter_points, use_second_graph,
                          llf, hops_bound, 0, efs[i], number_exper, number_of_threads);
    }
}