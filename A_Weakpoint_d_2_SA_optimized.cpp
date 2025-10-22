#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
#include<climits>
using namespace std;

struct Attack {
    int weapon;
    int box;
};

mt19937 rng(42);

vector<Attack> calculate_attacks(const vector<int>& box_order, int N, 
                                 const vector<int>& H, const vector<int>& C,
                                 const vector<vector<int>>& A) {
    vector<Attack> attacks;
    vector<bool> opened(N, false);
    vector<int> remaining_hardness = H;
    vector<int> weapon_durability = C;
    
    for (int box_idx : box_order) {
        if (opened[box_idx]) continue;
        
        int hardness = remaining_hardness[box_idx];
        
        // Collect available weapons with their damage values
        vector<pair<int, int>> weapon_uses; // (damage, weapon_id)
        for (int w = 0; w < N; w++) {
            if (opened[w] && weapon_durability[w] > 0) {
                weapon_uses.push_back({A[w][box_idx], w});
            }
        }
        
        // Sort by damage (descending)
        sort(weapon_uses.rbegin(), weapon_uses.rend());
        
        // Use best weapons first
        for (auto [damage, w] : weapon_uses) {
            if (hardness <= 0) break;
            
            int uses = min(weapon_durability[w], (hardness + damage - 1) / damage);
            for (int i = 0; i < uses; i++) {
                if (hardness <= 0) break;
                attacks.push_back({w, box_idx});
                hardness -= damage;
                weapon_durability[w]--;
            }
        }
        
        // Use bare hands for remaining hardness
        while (hardness > 0) {
            attacks.push_back({-1, box_idx});
            hardness--;
        }
        
        opened[box_idx] = true;
    }
    
    return attacks;
}

pair<int, vector<Attack>> evaluate_solution(const vector<int>& box_order, int N,
                                            const vector<int>& H, const vector<int>& C,
                                            const vector<vector<int>>& A) {
    auto attacks = calculate_attacks(box_order, N, H, C, A);
    int T = attacks.size();
    int total_hardness = 0;
    for (int h : H) total_hardness += h;
    int score = total_hardness - T + 1;
    return {-score, attacks}; // Return negative for minimization
}

vector<int> neighbor_swap(const vector<int>& order) {
    vector<int> new_order = order;
    int N = order.size();
    uniform_int_distribution<int> dist(0, N - 1);
    int i = dist(rng);
    int j = dist(rng);
    swap(new_order[i], new_order[j]);
    return new_order;
}

vector<int> neighbor_insert(const vector<int>& order) {
    vector<int> new_order = order;
    int N = order.size();
    uniform_int_distribution<int> dist(0, N - 1);
    int i = dist(rng);
    int j = dist(rng);
    int box = new_order[i];
    new_order.erase(new_order.begin() + i);
    new_order.insert(new_order.begin() + j, box);
    return new_order;
}

vector<int> neighbor_reverse(const vector<int>& order) {
    vector<int> new_order = order;
    int N = order.size();
    uniform_int_distribution<int> dist(0, N - 1);
    int i = dist(rng);
    int j = dist(rng);
    if (i > j) swap(i, j);
    reverse(new_order.begin() + i, new_order.begin() + j + 1);
    return new_order;
}

pair<vector<int>, vector<Attack>> simulated_annealing(int N, const vector<int>& H,
                                                       const vector<int>& C,
                                                       const vector<vector<int>>& A,
                                                       double time_limit = 1.8) {
    auto start_time = chrono::high_resolution_clock::now();
    
    // Try multiple initial solutions and pick the best
    vector<vector<int>> initial_solutions;
    
    // Solution 1: sort by hardness (ascending)
    vector<int> sol1(N);
    for (int i = 0; i < N; i++) sol1[i] = i;
    sort(sol1.begin(), sol1.end(), [&](int a, int b) { return H[a] < H[b]; });
    initial_solutions.push_back(sol1);
    
    // Solution 2: sort by weapon utility (total damage it can deal)
    vector<int> sol2(N);
    for (int i = 0; i < N; i++) sol2[i] = i;
    sort(sol2.begin(), sol2.end(), [&](int a, int b) {
        long long util_a = 0, util_b = 0;
        for (int j = 0; j < N; j++) {
            util_a += A[a][j];
            util_b += A[b][j];
        }
        util_a *= C[a];
        util_b *= C[b];
        return util_a > util_b;
    });
    initial_solutions.push_back(sol2);
    
    // Solution 3: random
    vector<int> sol3(N);
    for (int i = 0; i < N; i++) sol3[i] = i;
    shuffle(sol3.begin(), sol3.end(), rng);
    initial_solutions.push_back(sol3);
    
    // Pick best initial solution
    vector<int> current_order = initial_solutions[0];
    int best_init_cost = INT_MAX;
    for (auto& sol : initial_solutions) {
        auto [cost, _] = evaluate_solution(sol, N, H, C, A);
        if (cost < best_init_cost) {
            best_init_cost = cost;
            current_order = sol;
        }
    }
    cerr << "Best initial cost: " << -best_init_cost << endl;
    
    auto [current_cost, current_attacks] = evaluate_solution(current_order, N, H, C, A);
    
    vector<int> best_order = current_order;
    int best_cost = current_cost;
    vector<Attack> best_attacks = current_attacks;
    
    // SA parameters with adaptive cooling
    double T = 2000.0;
    double T_min = 0.001;
    double alpha = 0.99995;  // Slower cooling
    
    uniform_real_distribution<double> prob_dist(0.0, 1.0);
    uniform_int_distribution<int> neighbor_dist(0, 2);
    
    int iterations = 0;
    int accepts = 0;
    int improves = 0;
    
    while (T > T_min) {
        auto current_time = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(current_time - start_time).count();
        if (elapsed >= time_limit) break;
        
        iterations++;
        
        // Adaptive neighborhood selection
        vector<int> neighbor;
        double r = prob_dist(rng);
        if (T > 500.0) {
            // High temperature: more exploration (prefer insert/reverse)
            if (r < 0.3) neighbor = neighbor_swap(current_order);
            else if (r < 0.65) neighbor = neighbor_insert(current_order);
            else neighbor = neighbor_reverse(current_order);
        } else {
            // Low temperature: more exploitation (prefer swap)
            if (r < 0.6) neighbor = neighbor_swap(current_order);
            else if (r < 0.8) neighbor = neighbor_insert(current_order);
            else neighbor = neighbor_reverse(current_order);
        }
        
        auto [neighbor_cost, neighbor_attacks] = evaluate_solution(neighbor, N, H, C, A);
        
        // Accept or reject
        int delta = neighbor_cost - current_cost;
        
        if (delta < 0 || prob_dist(rng) < exp(-delta / T)) {
            current_order = neighbor;
            current_cost = neighbor_cost;
            current_attacks = neighbor_attacks;
            accepts++;
            
            if (current_cost < best_cost) {
                best_order = current_order;
                best_cost = current_cost;
                best_attacks = current_attacks;
                improves++;
                cerr << "New best: " << -best_cost << " at iter " << iterations << endl;
            }
        }
        
        T *= alpha;
    }
    
    cerr << "Iterations: " << iterations << endl;
    cerr << "Accepts: " << accepts << " (" << 100.0*accepts/iterations << "%)" << endl;
    cerr << "Improvements: " << improves << endl;
    cerr << "Final Score: " << -best_cost << endl;
    
    return {best_order, best_attacks};
}

int main() {
    int N;
    cin >> N;
    
    vector<int> H(N), C(N);
    for (int i = 0; i < N; i++) cin >> H[i];
    for (int i = 0; i < N; i++) cin >> C[i];
    
    vector<vector<int>> A(N, vector<int>(N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cin >> A[i][j];
        }
    }
    
    auto [best_order, best_attacks] = simulated_annealing(N, H, C, A);
    
    // Output attacks
    for (const auto& attack : best_attacks) {
        cout << attack.weapon << " " << attack.box << "\n";
    }
    
    return 0;
}