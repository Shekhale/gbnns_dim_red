#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
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
#include <cassert>

#include <algorithm>

#include <limits>
#include <sys/time.h>

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else
#include <x86intrin.h>
#endif

#define USE_AVX
#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif

using namespace std;

float EPS = 1e-10;

struct neighbor {
    uint32_t number;
    float dist;

    size_t operator()(const neighbor &n) const {
        size_t x = std::hash<uint32_t>()(n.number);

        return x;
    }
};

bool operator<(const neighbor& x, const neighbor& y)
{
    return x.dist < y.dist;
}


// reads 0 <= d < 4 floats as __m128
static inline __m128 masked_read (int d, const float *x)
{
//    assert (0 <= d && d < 4);
    __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
    switch (d) {
        case 3:
            buf[2] = x[2];
        case 2:
            buf[1] = x[1];
        case 1:
            buf[0] = x[0];
    }
    return _mm_load_ps (buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}


class Metric {
public:
    virtual float Dist(const float *x, const float *y, size_t d) = 0;
};


class LikeL2Metric : public Metric {
public:
    float Dist(const float *x, const float *y, size_t d) {
        float res = 0;
        for (int i = 0; i < d; ++i) {
            res += pow(*x - *y, 2);
            ++x;
            ++y;
        }
        return res;
    }
};


class L2Metric : public Metric {
public:
    float Dist(const float *pVect1, const float *pVect2, size_t qty) {
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty4 = qty >> 2;
        const float *pEnd1 = pVect1 + (qty4 << 2);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    };
};


class Angular : public Metric {
public:
    float Dist(const float *x, const float *y, size_t d) {
        __m256 msum1 = _mm256_setzero_ps();

        while (d >= 8) {
            __m256 mx = _mm256_loadu_ps (x); x += 8;
            __m256 my = _mm256_loadu_ps (y); y += 8;
            msum1 = _mm256_add_ps (msum1, _mm256_mul_ps (mx, my));
            d -= 8;
        }

        __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
        msum2 +=       _mm256_extractf128_ps(msum1, 0);

        if (d >= 4) {
            __m128 mx = _mm_loadu_ps (x); x += 4;
            __m128 my = _mm_loadu_ps (y); y += 4;
            msum2 = _mm_add_ps (msum2, _mm_mul_ps (mx, my));
            d -= 4;
        }

        if (d > 0) {
            __m128 mx = masked_read (d, x);
            __m128 my = masked_read (d, y);
            msum2 = _mm_add_ps (msum2, _mm_mul_ps (mx, my));
        }

        msum2 = _mm_hadd_ps (msum2, msum2);
        msum2 = _mm_hadd_ps (msum2, msum2);
        return  -_mm_cvtss_f32 (msum2);
    };
};


int FindGraphAverageDegree(vector< vector <uint32_t>> &graph) {
    double ans = 0;
    int n = graph.size();
    for (int i=0; i < n; ++i) {
        ans += graph[i].size();
    }
    return ans / n;
}


template<typename T>
void readXvec(std::ifstream &in, T *data, const size_t d, const size_t n = 1)
{
    uint32_t dim = d;
    for (size_t i = 0; i < n; i++) {
        in.read((char *) &dim, sizeof(uint32_t));
        if (dim != d) {
            std::cout << "file error\n";
            std::cout << "dim " << dim << ", d " << d << std::endl;
            std::cout << "our fault\n";

            exit(1);
        }
        in.read((char *) (data + i * dim), dim * sizeof(T));
    }
}


template<typename T>
void writeXvec(std::ofstream &out, T *data, const size_t d, const size_t n = 1)
{
    const uint32_t dim = d;
    for (size_t i = 0; i < n; i++) {
        out.write((char *) &dim, sizeof(uint32_t));
        out.write((char *) (data + i * dim), dim * sizeof(T));
    }
}

void write_edges(const char *location, const std::vector<std::vector<uint32_t>> &edges) {
    std::cout << "Saving edges to " << location << std::endl;
    std::ofstream output(location, std::ios::binary);

    for (uint32_t i = 0; i < edges.size(); i++) {
        const uint32_t *data = edges[i].data();
        uint32_t size = edges[i].size();

        output.write((char *) &size, sizeof(uint32_t));
        output.write((char *) data, sizeof(uint32_t) * size);
    }
}


vector<std::vector<uint32_t>> load_edges(string location, uint32_t n, string edges_name) {
//vector<std::vector<uint32_t>> load_edges(const char *location, std::vector<std::vector<uint32_t>> edges) {
    // std::cout << "Loading edges from " << location << std::endl;
    std::vector<std::vector<uint32_t>> edges(n);
    const char *location_char = location.c_str();
    std::ifstream input(location_char, std::ios::binary);
//    std::ifstream input(location, std::ios::binary);

    uint32_t size;
    for (int i = 0; i < edges.size(); i++) {
        input.read((char *) &size, sizeof(uint32_t));

        vector<uint32_t> vec(size);
        uint32_t *data = vec.data();
        input.read((char *) data, sizeof(uint32_t)*size);
        for (int j = 0; j < size; ++j) {
            edges[i].push_back(vec[j]);
        }
    }
    cout <<  edges_name + " " << FindGraphAverageDegree(edges) << endl;
    return edges;
}


vector<float> create_uniform_data(int N, int d, std::mt19937 random_gen) {
    vector<float> ds(N*d);
    normal_distribution<float> norm_distr(0, 1);
    for (int i=0; i < N; ++i) {
        vector<float> point(d);
        float norm_coeff = 0;
        for (int j=0; j < d; ++j) {
            point[j] = norm_distr(random_gen);
            norm_coeff += point[j] * point[j];
        }
        norm_coeff = pow(norm_coeff, 0.5);
        for (int j=0; j < d; ++j) {
            ds[i * d + j] = point[j] / norm_coeff;
        }
    }
    return  ds;
}

vector<uint32_t> get_truth(vector<float> ds, vector<float> query, int N, int d, int N_q, Metric *metric) {
    vector<uint32_t> truth(N_q);
#pragma omp parallel for
    for (uint32_t i=0; i < N_q; ++i) {
        const float* tendered_d = ds.data();
        const float* goal = query.data() + i*d;
        float dist = metric->Dist(tendered_d, goal, d);
        float new_dist = dist;
        uint32_t tendered_num = 0;
        for (uint32_t j=1; j<N; ++j) {
            tendered_d = ds.data() + j * d;
            new_dist = metric->Dist(tendered_d, goal, d);
            if (new_dist < dist) {
                dist = new_dist;
                tendered_num = j;
            }
        }
        truth[i] = tendered_num;
    }
    return truth;
}

vector< vector <uint32_t>> CutKNNbyThreshold(vector< vector <uint32_t>> &knn, vector<float> &ds, float thr, int N, int d,
                                  Metric *metric) {
    vector< vector <uint32_t>> knn_cut(N);
#pragma omp parallel for
    for (int i=0; i < N; ++i) {
        const float* point_i = ds.data() + i*d;
        for (int j=0; j < knn[i].size(); ++j) {
            int cur = knn[i][j];
            const float *point_cur = ds.data() + cur*d;
            if (metric->Dist(point_i, point_cur, d) < thr) {
                knn_cut[i].push_back(cur);
            }
        }
    }
    return  knn_cut;
}

vector< vector <uint32_t>> CutKNNbyK(vector< vector <uint32_t>> &knn, const float* ds, int knn_size, int N, int d,
                                             Metric *metric) {
    vector< vector <uint32_t>> knn_cut(N);
    bool small_size = false;
#pragma omp parallel for
    for (int i=0; i < N; ++i) {
        vector<neighbor> neigs;
        const float* point_i = ds + i*d;
        for (int j=0; j < knn[i].size(); ++j) {
            int cur = knn[i][j];
            const float *point_cur = ds + cur*d;
            neighbor neig{cur, metric->Dist(point_i, point_cur, d)};
            neigs.push_back(neig);
        }
        if (not small_size and knn_size > knn[i].size()) {
            cout << "Size knn less than you want" << endl;
            cout << knn[i].size() << endl;
            //exit(1);
            small_size = true;
        }

        sort(neigs.begin(), neigs.end());
        int cur_size = knn_size;
        if (knn[i].size() < cur_size) {
			cur_size = knn[i].size();
		}
        for (int j=0; j < cur_size; ++j) {
            knn_cut[i].push_back(neigs[j].number);
        }
    }
    return  knn_cut;
}


vector< vector <uint32_t>> CutKL(vector< vector <uint32_t>> &kl, int l, int N, vector< vector <uint32_t>> &knn) {
    vector< vector <uint32_t>> kl_cut(N);
    #pragma omp parallel for
    for (int i=0; i < N; ++i) {
        if (l > kl[i].size()) {
            cout << "Graph have less edges that you want" << endl;
            exit(1);
        }
        vector <uint32_t> kl_sh = kl[i];
        random_shuffle(kl_sh.begin(), kl_sh.end());
        int it = 0;
        while (kl_cut[i].size() < l and it < kl_sh.size()) {
            if (find(knn[i].begin(), knn[i].end(), kl_sh[it]) == knn[i].end()) {
                kl_cut[i].push_back(kl_sh[it]);
            }
            ++it;
        }
    }
    return  kl_cut;
}


int FindGraphMaxDegree(vector< vector <uint32_t>> &graph) {
    int max = 0;
    int n = graph.size();
    for (int i=0; i < n; ++i) {
        if (max < graph[i].size()) {
            max = graph[i].size();
        }
    }
    return max;
}


inline bool FileExist (std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}

vector< vector<uint32_t> > GraphMerge(vector< vector<uint32_t> > &graph_f, vector< vector<uint32_t> > &graph_s) {
    int n = graph_f.size();
    vector <vector<uint32_t> > union_graph(n);
#pragma omp parallel for
    for (int i=0; i < n; ++i) {
        for (int j =0; j < graph_f[i].size(); ++j) {
            union_graph[i].push_back(graph_f[i][j]);
        }
        for (int j =0; j < graph_s[i].size(); ++j) {
            if (find(union_graph[i].begin(), union_graph[i].end(), graph_s[i][j]) == union_graph[i].end()) {
                union_graph[i].push_back(graph_s[i][j]);
            }
        }
    }

    return union_graph;
}


vector< vector<uint32_t> > AddReverseEdgesForGD(vector< vector<uint32_t> > &gd_graph, const float* ds,
                               int M,  size_t N, size_t d, Metric *metric) {
//// NAIVE
//    vector< vector<uint32_t> > reverse_graph(N);
//    for (uint32_t i=0; i < N; ++i) {
//        for (uint32_t j=0; j < gd_graph[i].size(); ++j) {
//            if (reverse_graph[gd_graph[i][j]].size() < 2 * M) {
//                reverse_graph[gd_graph[i][j]].push_back(i);
//            }
//        }
//    }
//    gd_graph = GraphMerge(gd_graph, reverse_graph);

//// SMART

    vector< vector<uint32_t> > reverse_graph(N);
    for (uint32_t i=0; i < N; ++i) {
        for (uint32_t j=0; j < gd_graph[i].size(); ++j) {
            reverse_graph[gd_graph[i][j]].push_back(i);
        }
    }
    for (uint32_t i=0; i < N; ++i) {
        int upper_bound = M - reverse_graph[i].size();
        int reverse_threshold = min(upper_bound, M/2);
//            int reverse_threshold = static_cast<int>(M/2);
//            if (reverse_graph[i].size() < M) {
        if (reverse_threshold > 0) {
            for (uint32_t j=0; j < gd_graph[i].size(); ++j) {
                uint32_t cand = gd_graph[i][j];
                if (gd_graph[cand].size() < 2 * M) {
                    if (find(gd_graph[cand].begin(), gd_graph[cand].end(), i) == gd_graph[cand].end()) {
                        gd_graph[cand].push_back(i);
                        reverse_threshold -= 1;
                        if (reverse_threshold <= 0) {
                            break;
                        }
                    }
                }
            }
        }
    }

    return gd_graph;
}


void check_const_degree(vector< vector<uint32_t> > &graph) {
    if (graph.size() == 0) {
        return;
    }
    int degree = graph[0].size();
    bool failed = false;
#pragma omp parallel for
    for (uint32_t i=0; i < graph.size(); ++i) {
        if (graph[i].size() != degree) {
            failed = true;
        }
    }
    if (failed) {
        cout << " Graph degree is not constant " << endl;
    }
}


vector< vector<uint32_t> > GetConstantDegreeForGD(vector< vector<uint32_t> > &graph, const float* ds,
                               vector< vector<uint32_t> > &gd_graph,
                               int M,  size_t N, size_t d, Metric *metric) {

#pragma omp parallel for
    for (uint32_t i=0; i < N; ++i) {
        if (gd_graph[i].size() < 2 * M) {
            for (uint32_t j=1; j < graph[i].size(); ++j) {
                if (find(gd_graph[i].begin(), gd_graph[i].end(), graph[i][j]) == gd_graph[i].end()) {
                    gd_graph[i].push_back(graph[i][j]);
                    if (gd_graph[i].size() == 2 * M) {
                        break;
                    }
                }
            }
        }
    }
    check_const_degree(gd_graph);
    return gd_graph;
}



vector< vector<uint32_t> > FillGraphToConstantDegree(vector< vector<uint32_t> > &graph,
                               vector< vector<uint32_t> > &wide_graph, int degree_needed) {

    int max_degree = 0;
    int N = graph.size();
    for (uint32_t i=0; i < N; ++i) {
        if (graph[i].size() > max_degree) {
            max_degree = graph[i].size();
        }
    }
    if (degree_needed < max_degree) {
        degree_needed = max_degree;
    }

#pragma omp parallel for
    for (uint32_t i=0; i < N; ++i) {
        if (graph[i].size() < degree_needed) {
            for (uint32_t j=0; j < wide_graph[i].size(); ++j) {
                if (find(graph[i].begin(), graph[i].end(), wide_graph[i][j]) == graph[i].end()) {
                    graph[i].push_back(wide_graph[i][j]);
                    if (graph[i].size() == degree_needed) {
                        break;
                    }
                }
            }
        }
    }
    check_const_degree(graph);
    return graph;
}


vector< vector<uint32_t> > hnswlikeGD(vector< vector<uint32_t> > &graph, const float* ds,
                              int M,  size_t N, size_t d, Metric *metric, bool reverse,
                              bool need_const_degree) {

    vector< vector<uint32_t> > gd_graph(N);

//    int edge = 5;
    int edge = static_cast<int>(M/2);
#pragma omp parallel for
    for (uint32_t i=0; i < N; ++i) {
        vector<neighbor> neighbors;
        const float* point_i = ds + i * d;
        for (uint32_t j=0; j < graph[i].size(); ++j) {
            const float* point_cur = ds + graph[i][j] * d;
            float dist_i = metric->Dist(point_i, point_cur, d);
            if (dist_i > EPS) {
                neighbor neig{graph[i][j], dist_i};
                neighbors.push_back(neig);
            }
        }
        sort(neighbors.begin(), neighbors.end());
        gd_graph[i].push_back(neighbors[0].number);
        for (uint32_t j=1; j < neighbors.size(); ++j) {
            const float* point_pre = ds + neighbors[j].number * d;
            bool good = true;
            for (uint32_t l=0; l < gd_graph[i].size(); ++l) {
                const float* point_alr = ds + gd_graph[i][l] * d;
                if (metric->Dist(point_pre, point_i, d) + EPS > metric->Dist(point_pre, point_alr, d)) {
                    good = false;
                    break;
                }
            }
            if (good) {
				gd_graph[i].push_back(neighbors[j].number);
			}
            if (gd_graph[i].size() == M) {
                break;
            }
        }
        for (uint32_t j=0; j < edge; ++j) {
            if (find(gd_graph[i].begin(), gd_graph[i].end(), neighbors[j].number) == gd_graph[i].end()) {
                gd_graph[i].push_back(neighbors[j].number);
            }
		}
    }

    if (reverse) {
        gd_graph = AddReverseEdgesForGD(gd_graph, ds, M, N, d, metric);
    }

    if (need_const_degree) {
        gd_graph = GetConstantDegreeForGD(graph, ds, gd_graph, M, N, d, metric);
    }

    return gd_graph;
}


std::vector<std::string> SplitString(const std::string& str, char delimiter) {

    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}


std::map<std::string, std::string> AddMapFromStr(std::string str, std::map<std::string, std::string> params_map,
                                                 std::string global_key) {
    char delimiter(' ');
    std::vector<string> str_sep = SplitString(str, delimiter);
    if (str_sep.size() > 0 and str_sep[0] == global_key and str_sep.size() == 3) {
        params_map[str_sep[1]] = str_sep[2];
    }
    return params_map;
}


std::map<std::string, std::string> ReadSearchParams(std::string file_name, std::string database_name) {
    std::map<std::string, std::string> params_map;
    std::ifstream file(file_name);
    std::string str;
    while (std::getline(file, str)) {
        params_map = AddMapFromStr(str, params_map, database_name);
    }

    return params_map;
}


vector<int> VectorFromString(string str) {
    vector<int> ans;
    vector<string> str_sep = SplitString(str, ',');
    for (int i = 0; i < str_sep.size(); ++i) {
        ans.push_back(atoi(str_sep[i].c_str()));
    }

    return ans;
}


void PrintFirst(vector<float> &ds, int start, int len) {
    for (int i = start; i < start + len; ++i) {
        cout << ds[i] << ' ';

    }
    cout << endl;
}


void net_layer_compute(const float* matrix, const float* input, vector<float> &output, bool activation,
                       int step, int d_in, int d_out, Metric *ang) {
    for (int i = 0; i < d_out; ++i) {
        output[i] -=  ang->Dist(matrix + i * step, input, d_in);
        output[i] += *(matrix + i * step + step - 1);
        if ( activation && output[i] < 0) {
            output[i] = 0;
        }
    }
}


void normalize_vector(vector<float> &input, const float* zeros, int d, Metric *l2) {
    float norm = l2->Dist(input.data(), zeros, d);
//    float norm = 0;
//    for (int i = 0; i < d; ++i) {
//        norm += input[i] * input[i];
//    }
    norm = sqrt(norm);
    for (int i = 0; i < d; ++i) {
        input[i] /= norm;
    }
}


void get_low_query_from_net_3(const float* matrix_1,  const float* matrix_2, const float* matrix_3,
                              const float* query, vector<float> &ans, const float* zeros,
                              size_t d, size_t d_hidden, size_t d_hidden_2, size_t d_low, Metric *ang, Metric *l2) {

    vector<float> hidden_layer(d_hidden);
    net_layer_compute(matrix_1, query, hidden_layer, true, d + 1, d, d_hidden, ang);

    vector<float> hidden_2_layer(d_hidden_2);
    net_layer_compute(matrix_2, hidden_layer.data(), hidden_2_layer, true, d_hidden + 1, d_hidden, d_hidden_2, ang);

    net_layer_compute(matrix_3, hidden_2_layer.data(), ans, false, d_hidden_2 + 1, d_hidden_2, d_low, ang);

    normalize_vector(ans, zeros, d_low, l2);

}


template<typename T>
vector<T> loadXvecs(string data_path, const size_t d, const size_t n = 1) {
    vector<T> data(n * d);
    const char *data_path_char = data_path.c_str();
    std::ifstream data_input(data_path_char, std::ios::binary);
    readXvec<T>(data_input, data.data(), d, n);

    return data;
}