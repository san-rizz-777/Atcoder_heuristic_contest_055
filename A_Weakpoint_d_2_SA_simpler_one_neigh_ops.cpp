#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cmath>
#include <chrono>
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

// Simple swap neighborhood
vector<int> neighbor_swap(const vector<int>& order) {
    vector<int> new_order = order;
    int N = order.size();
    uniform_int_distribution<int> dist(0, N - 1);
    int i = dist(rng);
    int j = dist(rng);
    swap(new_order[i], new_order[j]);
    return new_order;
}

pair<vector<int>, vector<Attack>> simulated_annealing(int N, const vector<int>& H,
                                                       const vector<int>& C,
                                                       const vector<vector<int>>& A,
                                                       double time_limit = 1.8) {
    auto start_time = chrono::high_resolution_clock::now();
    
    // Simple initial solution: sort by hardness (ascending)
    vector<int> current_order(N);
    for (int i = 0; i < N; i++) current_order[i] = i;
    sort(current_order.begin(), current_order.end(), 
         [&](int a, int b) { return H[a] < H[b]; });
    
    auto [current_cost, current_attacks] = evaluate_solution(current_order, N, H, C, A);
    
    vector<int> best_order = current_order;
    int best_cost = current_cost;
    vector<Attack> best_attacks = current_attacks;
    
    // Simple SA parameters
    double T = 1000.0;
    double T_min = 0.01;
    double alpha = 0.9995;
    
    uniform_real_distribution<double> prob_dist(0.0, 1.0);
    
    int iterations = 0;
    int accepts = 0;
    int improves = 0;
    
    while (T > T_min) {
        auto current_time = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(current_time - start_time).count();
        if (elapsed >= time_limit) break;
        
        iterations++;
        
        // Simple neighborhood: only swap
        vector<int> neighbor = neighbor_swap(current_order);
        
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
            }
        }
        
        T *= alpha;
    }
    
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