#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <climits>
#include <queue>
using namespace std;

struct Move {
    int weapon;
    int box;
};

mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

struct AnnealingSystem {
    double T_start, T_end;
    double current_temp;
    bool is_maximize;
    uniform_real_distribution<double> dist;

    AnnealingSystem(double delta_high, double delta_low, double p_high = 0.9, double p_low = 1e-4, bool maximize = false)
        : is_maximize(maximize), dist(0.0, 1.0) {
        T_start = -delta_high / log(p_high);
        T_end = -delta_low / log(p_low);
        if (T_end <= 0) T_end = 1e-12;
        current_temp = T_start;
    }

    bool accept(double curr_score, double next_score) {
        if (is_maximize) {
            if (next_score >= curr_score) return true;
            double diff = curr_score - next_score;
            return dist(rng) < exp(-diff / current_temp);
        } else {
            if (next_score <= curr_score) return true;
            double diff = next_score - curr_score;
            return dist(rng) < exp(-diff / current_temp);
        }
    }

    void update(double elapsed, double total) {
        if (elapsed <= 0) {
            current_temp = T_start;
            return;
        }
        if (elapsed >= total) {
            current_temp = T_end;
            return;
        }
        double ratio = elapsed / total;
        current_temp = T_start * pow(T_end / T_start, ratio);
    }
};

struct WeightedChoice {
    vector<vector<double>> probs;
    vector<vector<int>> aliases;
    vector<int> sizes;

    void init(const vector<double>& weights, int idx) {
        int sz = weights.size();
        if (idx >= (int)probs.size()) {
            probs.resize(idx + 1);
            aliases.resize(idx + 1);
            sizes.resize(idx + 1);
        }
        sizes[idx] = sz;
        probs[idx].assign(sz, 0.0);
        aliases[idx].assign(sz, 0);
        if (sz == 0) return;

        vector<double> scaled(sz);
        vector<int> small_list, large_list;
        double total = 0;
        for (double w : weights) total += w;
        for (int i = 0; i < sz; i++) scaled[i] = weights[i] / total * sz;

        for (int i = 0; i < sz; i++) {
            if (scaled[i] < 1.0) small_list.push_back(i);
            else large_list.push_back(i);
        }

        while (!small_list.empty() && !large_list.empty()) {
            int s = small_list.back(); small_list.pop_back();
            int l = large_list.back(); large_list.pop_back();
            probs[idx][s] = scaled[s];
            aliases[idx][s] = l;
            scaled[l] -= 1.0 - scaled[s];
            if (scaled[l] < 1.0) small_list.push_back(l);
            else large_list.push_back(l);
        }

        for (int i : large_list) probs[idx][i] = 1.0, aliases[idx][i] = i;
        for (int i : small_list) probs[idx][i] = 1.0, aliases[idx][i] = i;
    }

    int get(int idx) {
        int pos = rng() % sizes[idx];
        double r = (rng() & 0xFFFFFFFF) / double(0x100000000ULL);
        return r < probs[idx][pos] ? pos : aliases[idx][pos];
    }
};

bool check_cycle(int start, int end, const vector<vector<pair<int, int>>>& graph, 
                 vector<int>& visited, int& cycle_id) {
    if (start == end) return false;
    
    queue<int> q;
    cycle_id++;
    q.push(end);
    visited[end] = cycle_id;
    
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        for (auto [next, _] : graph[node]) {
            if (next == start) return false;
            if (visited[next] != cycle_id) {
                q.push(next);
                visited[next] = cycle_id;
            }
        }
    }
    return true;
}

void solve() {
    auto start_time = chrono::high_resolution_clock::now();
    
    int N;
    cin >> N;
    
    int total_hardness = 0;
    vector<int> H(N);
    vector<pair<int, int>> hardness_pairs;
    for (int i = 0; i < N; i++) {
        cin >> H[i];
        total_hardness += H[i];
        hardness_pairs.push_back({H[i], i});
    }
    
    vector<int> orig_H = H;
    vector<int> C(N);
    for (int i = 0; i < N; i++) cin >> C[i];
    
    vector<vector<int>> A(N, vector<int>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cin >> A[i][j];
        }
    }
    
    WeightedChoice sampler1, sampler2, sampler3;
    double power = 1.5;
    
    for (int i = 0; i < N; i++) {
        vector<double> weights;
        for (int j = 0; j < N; j++) {
            weights.push_back(pow((double)A[i][j] - 1, power));
        }
        sampler1.init(weights, i);
    }
    
    for (int i = 0; i < N; i++) {
        vector<double> weights;
        for (int j = 0; j < N; j++) {
            weights.push_back(pow((double)A[j][i] - 1, power));
        }
        sampler2.init(weights, i);
    }
    
    vector<double> h_weights;
    for (int i = 0; i < N; i++) {
        h_weights.push_back(pow((double)H[i], power));
    }
    sampler3.init(h_weights, 0);
    
    vector<vector<pair<int, int>>> edges(N);
    vector<int> in_degree(N);
    int current_score = total_hardness;
    
    vector<int> visited(N);
    int cycle_counter = 0;
    
    int retry_limit = 20;
    int best_score = 1e9;
    double time_limit = 1.99;
    
    AnnealingSystem scheduler((double)total_hardness * 0.003, (double)total_hardness * 0.00001);
    
    sort(hardness_pairs.rbegin(), hardness_pairs.rend());
    
    for (int iter = 0; iter < 80; iter++) {
        int target = hardness_pairs[iter].second;
        int retries = 0;
        while (H[target] <= 0 && retries < retry_limit) {
            target = sampler3.get(0);
            retries++;
        }
        if (retries >= retry_limit) continue;
        
        int weapon = sampler2.get(target);
        retries = 0;
        while ((weapon == target || C[weapon] == 0) && retries < retry_limit) {
            weapon = sampler2.get(target);
            retries++;
        }
        
        bool accepted = scheduler.accept(current_score, current_score - min(H[target], A[weapon][target]) + 1);
        if (!accepted || retries >= retry_limit) continue;
        
        int edge_idx = -1;
        for (int i = 0; i < (int)edges[weapon].size(); i++) {
            if (edges[weapon][i].first == target) edge_idx = i;
        }
        
        in_degree[target]++;
        C[weapon]--;
        
        if (edge_idx == -1) {
            if (!check_cycle(weapon, target, edges, visited, cycle_counter)) {
                in_degree[target]--;
                C[weapon]++;
            } else {
                edges[weapon].push_back({target, 1});
                current_score = current_score - min(H[target], A[weapon][target]) + 1;
                H[target] -= A[weapon][target];
            }
        } else {
            edges[weapon][edge_idx].second++;
            current_score = current_score - min(H[target], A[weapon][target]) + 1;
            H[target] -= A[weapon][target];
        }
    }
    
    int iterations = 0;
    while (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time).count() < time_limit * 1000) {
        unsigned long long rand_val = rng();
        int operation = rand_val % 10;
        iterations++;
        
        auto now = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(now - start_time).count();
        scheduler.update(elapsed, time_limit);
        
        best_score = min(best_score, current_score);
        
        if (operation <= 1) {
            unsigned long long r = rng();
            int sub_op = r % 10;
            
            if (sub_op <= 4) {
                int target = sampler3.get(0);
                int retries = 0;
                while (H[target] <= 0 && retries < retry_limit) {
                    target = sampler3.get(0);
                    retries++;
                }
                if (retries >= retry_limit) continue;
                
                int weapon = sampler2.get(target);
                retries = 0;
                while ((weapon == target || C[weapon] == 0) && retries < retry_limit) {
                    weapon = sampler2.get(target);
                    retries++;
                }
                
                bool accepted = scheduler.accept(current_score, current_score - min(H[target], A[weapon][target]) + 1);
                if (!accepted || retries >= retry_limit) continue;
                
                int edge_idx = -1;
                for (int i = 0; i < (int)edges[weapon].size(); i++) {
                    if (edges[weapon][i].first == target) edge_idx = i;
                }
                
                in_degree[target]++;
                C[weapon]--;
                
                if (edge_idx == -1) {
                    if (!check_cycle(weapon, target, edges, visited, cycle_counter)) {
                        in_degree[target]--;
                        C[weapon]++;
                    } else {
                        edges[weapon].push_back({target, 1});
                        current_score = current_score - min(H[target], A[weapon][target]) + 1;
                        H[target] -= A[weapon][target];
                    }
                } else {
                    edges[weapon][edge_idx].second++;
                    current_score = current_score - min(H[target], A[weapon][target]) + 1;
                    H[target] -= A[weapon][target];
                }
            } else {
                unsigned long long l = rng();
                int weapon = l % N;
                int retries = 0;
                while (edges[weapon].empty() && retries < retry_limit) {
                    l = rng();
                    weapon = l % N;
                    retries++;
                }
                if (retries >= retry_limit) continue;
                
                int sz = edges[weapon].size();
                unsigned long long r = rng();
                int edge_pos = r % sz;
                int orig_pos = edge_pos;
                int target = edges[weapon][edge_pos].first;
                
                bool accepted = scheduler.accept(current_score, current_score + min(A[weapon][target], max(A[weapon][target] + H[target], 0)) - 1);
                if (accepted) {
                    if (edges[weapon][orig_pos].second == 1) {
                        edges[weapon].erase(edges[weapon].begin() + orig_pos);
                    } else {
                        edges[weapon][orig_pos].second--;
                    }
                    C[weapon]++;
                    current_score = current_score + min(A[weapon][target], max(A[weapon][target] + H[target], 0)) - 1;
                    H[target] += A[weapon][target];
                    in_degree[target]--;
                }
            }
        } else {
            unsigned long long s = rng();
            int sub_op = s % 10;
            
            if (sub_op <= 6) {
                unsigned long long l = rng();
                int weapon = l % N;
                int retries = 0;
                while (edges[weapon].empty() && retries < retry_limit) {
                    l = rng();
                    weapon = l % N;
                    retries++;
                }
                if (retries >= retry_limit) continue;
                
                int new_target = sampler1.get(weapon);
                while (new_target == weapon) {
                    new_target = sampler1.get(weapon);
                }
                
                unsigned long long r2 = rng();
                int old_pos = r2 % (int)edges[weapon].size();
                int old_target = edges[weapon][old_pos].first;
                
                retries = 0;
                while ((new_target == old_target || H[new_target] <= 0) && retries < retry_limit) {
                    new_target = sampler1.get(weapon);
                    while (new_target == weapon) {
                        new_target = sampler1.get(weapon);
                    }
                    r2 = rng();
                    old_pos = r2 % (int)edges[weapon].size();
                    old_target = edges[weapon][old_pos].first;
                    retries++;
                }
                
                bool accepted = scheduler.accept(current_score, current_score + min(A[weapon][old_target], max(A[weapon][old_target] + H[old_target], 0)) - min(H[new_target], A[weapon][new_target]));
                if (!accepted || retries >= retry_limit) continue;
                
                int old_edge_idx = -1;
                for (int i = 0; i < (int)edges[weapon].size(); i++) {
                    if (edges[weapon][i].first == old_target) old_edge_idx = i;
                }
                if (old_edge_idx == -1) continue;
                
                bool removed = false;
                if (edges[weapon][old_edge_idx].second == 1) {
                    edges[weapon].erase(edges[weapon].begin() + old_edge_idx);
                    removed = true;
                } else {
                    edges[weapon][old_edge_idx].second--;
                }
                
                if (!check_cycle(weapon, new_target, edges, visited, cycle_counter)) {
                    if (removed) {
                        edges[weapon].push_back({old_target, 1});
                    } else {
                        edges[weapon][old_edge_idx].second++;
                    }
                } else {
                    int new_edge_idx = -1;
                    for (int i = 0; i < (int)edges[weapon].size(); i++) {
                        if (edges[weapon][i].first == new_target) new_edge_idx = i;
                    }
                    
                    if (new_edge_idx == -1) {
                        edges[weapon].push_back({new_target, 1});
                    } else {
                        edges[weapon][new_edge_idx].second++;
                    }
                    
                    in_degree[new_target]++;
                    in_degree[old_target]--;
                    current_score = current_score + min(A[weapon][old_target], max(A[weapon][old_target] + H[old_target], 0)) - min(H[new_target], A[weapon][new_target]);
                    H[old_target] += A[weapon][old_target];
                    H[new_target] -= A[weapon][new_target];
                }
            }
        }
    }
    
    cerr << "Iterations: " << iterations << endl;
    cerr << "Best score: " << best_score << endl;
    cerr << "Final score: " << current_score << endl;
    
    // Output solution
    queue<int> q;
    vector<int> visited_output(N);
    vector<pair<int, int>> result;
    
    for (int i = 0; i < N; i++) {
        if (in_degree[i] == 0) {
            q.push(i);
            visited_output[i] = 1;
        }
    }
    
    vector<int> temp_degree = in_degree;
    while (!q.empty()) {
        int box = q.front();
        q.pop();
        
        for (int i = 0; i < orig_H[box]; i++) {
            result.push_back({-1, box});
        }
        
        for (auto [target, count] : edges[box]) {
            temp_degree[target] -= count;
            if (visited_output[target] == 0 && temp_degree[target] == 0) {
                q.push(target);
                visited_output[target] = 1;
            }
            for (int i = 0; i < count; i++) {
                if (orig_H[target] <= 0) break;
                orig_H[target] -= A[box][target];
                result.push_back({box, target});
            }
        }
    }
    
    for (auto [w, b] : result) {
        cout << w << " " << b << "\n";
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    solve();
    return 0;
}