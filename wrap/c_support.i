%module c_support
%{
 /* Put header files here or function declarations like below */
 extern int get_graphs_and_search_tests(char transform_type, char dataset, int d_p, int d_low_p, int n_q_p, char val, int n_val, bool reverse_gd);
%}

// extern int get_graphs_and_tests(char file_char);
 extern int get_graphs_and_search_tests(char transform_type, char dataset, int d_p, int d_low_p, int n_q_p, char val, int n_val, bool reverse_gd);