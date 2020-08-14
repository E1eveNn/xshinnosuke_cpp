#include "toolkit.h"


void initialize_variables_grad(const initializer_list<Variable*>& l, bool is_training) {
	if (is_training) {
		for (auto v = l.begin(); v != l.end(); ++v) {
			if ((*v)->requires_grad) {
				(*v)->zero_grad();
			}
		}
	}
}

Eigen::MatrixXf& mat_log(Eigen::MatrixXf& mat) {
	int rows = mat.rows();
	int cols = mat.cols();
	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			mat(r, c) = log(mat(r, c));
		}
	}
	return mat;
}

template <class T>
vector<T*> topological_sort(T* inputs, T* outputs) {

	vector<T*> sorted_graph;
	vector<T*> outs{ outputs };
	vector<T*> ins{ inputs };

	unordered_map<T*, unordered_map<string, unordered_set<T*>> > G;
	while (ins.size() > 0)
	{
		T* n = ins.back();
		ins.pop_back();
		if (G.count(n) == 0) {
			unordered_map<string, unordered_set<T*>>
				m1{ {"in", unordered_set<T*>()},
					{"out", unordered_set<T*>()} };
			G[n] = m1;
		}
		for (auto m = n->out_bounds.begin(); m != n->out_bounds.end(); ++m) {
			if (G.count(*m) == 0) {
				unordered_map<string, unordered_set<T*>>
					m1{ {"in", unordered_set<T*>()},
						{"out", unordered_set<T*>()} };
				G[*m] = m1;
			}
			G[n]["out"].insert(*m);
			G[*m]["in"].insert(n);
			ins.push_back(*m);
		}
	}

	unordered_set<T*> S{ inputs };
	while (S.size() > 0)
	{
		T* n = S.end();
		S.erase(S.end());
		sorted_graph.push_back(n);
		if (std::find(outs.begin(), outs.end(), n) != outs.end()) {
			continue;
		}
		for (auto m = n->out_bounds.begin(); m != n->out_bounds.end(); ++m) {
			G[n]["out"].erase(*m);
			G[*m]["in"].erase(n);
			if (G[*m]["in"].size() == 0) {
				S.insert(*m);
			}
		}
	}
	return sorted_graph;
}