#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include <climits>
#include <queue>
using namespace std;

// Represents a single attack action (weapon index, box index)
struct Move {
    int weapon;
    int box;
};

// Random number generator seeded with current time
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

// Simulated Annealing temperature scheduler
// Controls the probability of accepting worse solutions during optimization
struct AnnealingSystem {
    double T_start, T_end;      // Initial and final temperature
    double current_temp;         // Current temperature
    bool is_maximize;            // Whether we're maximizing or minimizing
    uniform_real_distribution<double> dist;

    // Initialize temperature range based on expected score differences
    AnnealingSystem(double delta_high, double delta_low, double p_high = 0.9, double p_low = 1e-4, bool maximize = false)
        : is_maximize(maximize), dist(0.0, 1.0) {
        // Calculate temperatures to achieve desired acceptance probabilities
        T_start = -delta_high / log(p_high);
        T_end = -delta_low / log(p_low);
        if (T_end <= 0) T_end = 1e-12;
        current_temp = T_start;
    }

    // Decide whether to accept a new solution based on current temperature
    // Always accepts improvements, probabilistically accepts worse solutions
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

    // Update temperature based on elapsed time (exponential cooling schedule)
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
        // Exponential interpolation between T_start and T_end
        current_temp = T_start * pow(T_end / T_start, ratio);
    }
};

// Weighted random sampler using the Alias Method
// Allows efficient O(1) sampling from discrete probability distribution
struct WeightedChoice {
    vector<vector<double>> probs;    // Probability table for alias method
    vector<vector<int>> aliases;     // Alias table
    vector<int> sizes;               // Size of each distribution

    // Build alias method tables for a given weight distribution
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

        // Normalize weights and scale to [0, sz]
        vector<double> scaled(sz);
        vector<int> small_list, large_list;
        double total = 0;
        for (double w : weights) total += w;
        for (int i = 0; i < sz; i++) scaled[i] = weights[i] / total * sz;

        // Partition into small (<1) and large (>=1) probabilities
        for (int i = 0; i < sz; i++) {
            if (scaled[i] < 1.0) small_list.push_back(i);
            else large_list.push_back(i);
        }

        // Build alias table by pairing small and large probabilities
        while (!small_list.empty() && !large_list.empty()) {
            int s = small_list.back(); small_list.pop_back();
            int l = large_list.back(); large_list.pop_back();
            probs[idx][s] = scaled[s];
            aliases[idx][s] = l;
            scaled[l] -= 1.0 - scaled[s];
            if (scaled[l] < 1.0) small_list.push_back(l);
            else large_list.push_back(l);
        }

        // Handle remaining elements
        for (int i : large_list) probs[idx][i] = 1.0, aliases[idx][i] = i;
        for (int i : small_list) probs[idx][i] = 1.0, aliases[idx][i] = i;
    }

    // Sample an index according to the weighted distribution
    int get(int idx) {
        int pos = rng() % sizes[idx];
        double r = (rng() & 0xFFFFFFFF) / double(0x100000000ULL);
        return r < probs[idx][pos] ? pos : aliases[idx][pos];
    }
};

// Check if adding edge (start -> end) would create a cycle in the graph
// Uses BFS to detect if there's already a path from end to start
bool check_cycle(int start, int end, const vector<vector<pair<int, int>>>& graph, 
                 vector<int>& visited, int& cycle_id) {
    if (start == end) return false;  // Self-loop would be a cycle
    
    queue<int> q;
    cycle_id++;  // Use unique ID to avoid resetting visited array
    q.push(end);
    visited[end] = cycle_id;
    
    // BFS from end node
    while (!q.empty()) {
        int node = q.front();
        q.pop();
        for (auto [next, _] : graph[node]) {
            if (next == start) return false;  // Found path end -> start, would create cycle
            if (visited[next] != cycle_id) {
                q.push(next);
                visited[next] = cycle_id;
            }
        }
    }
    return true;  // No cycle detected
}

void solve() {
    auto start_time = chrono::high_resolution_clock::now();
    
    // Read input
    int N;
    cin >> N;
    
    int total_hardness = 0;
    vector<int> H(N);  // Hardness of each box
    vector<pair<int, int>> hardness_pairs;  // (hardness, box_id) for sorting
    for (int i = 0; i < N; i++) {
        cin >> H[i];
        total_hardness += H[i];
        hardness_pairs.push_back({H[i], i});
    }
    
    vector<int> orig_H = H;  // Keep original hardness for output phase
    vector<int> C(N);        // Durability of each weapon
    for (int i = 0; i < N; i++) cin >> C[i];
    
    vector<vector<int>> A(N, vector<int>(N));  // A[i][j] = damage when weapon i attacks box j
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cin >> A[i][j];
        }
    }
    
    // Initialize three weighted samplers with power law distribution
    WeightedChoice sampler1, sampler2, sampler3;
    double power = 1.5;  // Exponent for weight calculation
    
    // sampler1[i]: For weapon i, sample target boxes weighted by damage
    for (int i = 0; i < N; i++) {
        vector<double> weights;
        for (int j = 0; j < N; j++) {
            weights.push_back(pow((double)A[i][j] - 1, power));
        }
        sampler1.init(weights, i);
    }
    
    // sampler2[i]: For box i, sample weapons weighted by their damage to box i
    for (int i = 0; i < N; i++) {
        vector<double> weights;
        for (int j = 0; j < N; j++) {
            weights.push_back(pow((double)A[j][i] - 1, power));
        }
        sampler2.init(weights, i);
    }
    
    // sampler3: Sample boxes weighted by their hardness
    vector<double> h_weights;
    for (int i = 0; i < N; i++) {
        h_weights.push_back(pow((double)H[i], power));
    }
    sampler3.init(h_weights, 0);
    
    // Graph representation: edges[i] = list of (target_box, count) that weapon i attacks
    vector<vector<pair<int, int>>> edges(N);
    vector<int> in_degree(N);  // Number of incoming edges (dependencies) for each box
    int current_score = total_hardness;  // Current score (lower is better)
    
    // For cycle detection
    vector<int> visited(N);
    int cycle_counter = 0;
    
    int retry_limit = 20;
    int best_score = 1e9;
    double time_limit = 1.99;  // Stay under 2 second time limit
    
    // Initialize annealing system with temperature proportional to total hardness
    AnnealingSystem scheduler((double)total_hardness * 0.003, (double)total_hardness * 0.00001);
    
    // Phase 1: Greedy initialization - focus on hardest boxes first
    sort(hardness_pairs.rbegin(), hardness_pairs.rend());
    
    for (int iter = 0; iter < 80; iter++) {
        int target = hardness_pairs[iter].second;
        int retries = 0;
        
        // Find a box that still needs opening
        while (H[target] <= 0 && retries < retry_limit) {
            target = sampler3.get(0);
            retries++;
        }
        if (retries >= retry_limit) continue;
        
        // Find a weapon with good damage against this box
        int weapon = sampler2.get(target);
        retries = 0;
        while ((weapon == target || C[weapon] == 0) && retries < retry_limit) {
            weapon = sampler2.get(target);
            retries++;
        }
        
        // Check if annealing system accepts this move
        bool accepted = scheduler.accept(current_score, current_score - min(H[target], A[weapon][target]) + 1);
        if (!accepted || retries >= retry_limit) continue;
        
        // Check if edge already exists
        int edge_idx = -1;
        for (int i = 0; i < (int)edges[weapon].size(); i++) {
            if (edges[weapon][i].first == target) edge_idx = i;
        }
        
        in_degree[target]++;
        C[weapon]--;
        
        if (edge_idx == -1) {
            // New edge - check for cycle
            if (!check_cycle(weapon, target, edges, visited, cycle_counter)) {
                // Would create cycle, reject
                in_degree[target]--;
                C[weapon]++;
            } else {
                // No cycle, add edge
                edges[weapon].push_back({target, 1});
                current_score = current_score - min(H[target], A[weapon][target]) + 1;
                H[target] -= A[weapon][target];
            }
        } else {
            // Edge exists, increment count
            edges[weapon][edge_idx].second++;
            current_score = current_score - min(H[target], A[weapon][target]) + 1;
            H[target] -= A[weapon][target];
        }
    }
    
    // Phase 2: Simulated annealing optimization
    int iterations = 0;
    while (chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - start_time).count() < time_limit * 1000) {
        unsigned long long rand_val = rng();
        int operation = rand_val % 10;
        iterations++;
        
        // Update temperature based on elapsed time
        auto now = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(now - start_time).count();
        scheduler.update(elapsed, time_limit);
        
        best_score = min(best_score, current_score);
        
        // Operation Type 1 (20% probability): Add or remove edges
        if (operation <= 1) {
            unsigned long long r = rng();
            int sub_op = r % 10;
            
            // Sub-operation A (50%): Add new edge
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
            } 
            // Sub-operation B (50%): Remove existing edge
            else {
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
        } 
        // Operation Type 2 (80% probability): Replace edge target
        else {
            unsigned long long s = rng();
            int sub_op = s % 10;
            
            // Sub-operation A (70%): Replace an edge's target
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
                
                // Sample new target
                int new_target = sampler1.get(weapon);
                while (new_target == weapon) {
                    new_target = sampler1.get(weapon);
                }
                
                // Pick existing edge to replace
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
                
                // Remove old edge
                bool removed = false;
                if (edges[weapon][old_edge_idx].second == 1) {
                    edges[weapon].erase(edges[weapon].begin() + old_edge_idx);
                    removed = true;
                } else {
                    edges[weapon][old_edge_idx].second--;
                }
                
                // Try to add new edge (check cycle)
                if (!check_cycle(weapon, new_target, edges, visited, cycle_counter)) {
                    // Would create cycle, restore old edge
                    if (removed) {
                        edges[weapon].push_back({old_target, 1});
                    } else {
                        edges[weapon][old_edge_idx].second++;
                    }
                } else {
                    // No cycle, commit the change
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
    
    // Log statistics to stderr
    cerr << "Iterations: " << iterations << endl;
    cerr << "Best score: " << best_score << endl;
    cerr << "Final score: " << current_score << endl;
    
    // Phase 3: Extract solution using topological sort
    queue<int> q;
    vector<int> visited_output(N);
    vector<pair<int, int>> result;
    
    // Start with boxes that have no dependencies
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
        
        // First, use bare hands to open this box
        for (int i = 0; i < orig_H[box]; i++) {
            result.push_back({-1, box});
        }
        
        // Then use the weapon from this box to attack other boxes
        for (auto [target, count] : edges[box]) {
            temp_degree[target] -= count;
            // If target has no more dependencies, it can be processed
            if (visited_output[target] == 0 && temp_degree[target] == 0) {
                q.push(target);
                visited_output[target] = 1;
            }
            // Execute the attacks
            for (int i = 0; i < count; i++) {
                if (orig_H[target] <= 0) break;
                orig_H[target] -= A[box][target];
                result.push_back({box, target});
            }
        }
    }
    
    // Output the sequence of attacks
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