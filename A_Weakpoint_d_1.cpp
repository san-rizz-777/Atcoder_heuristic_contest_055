#include<bits/stdc++.h>

using namespace std;

int main()
{
    int n;
    cin >> n;
    int sum = 0;
    vector<int> h(n);      // hardness of box
    for(int i = 0; i < n; i++)   
    {
        cin >> h[i];
        sum += h[i];
    }   

    vector<int> c(n);  // durability of weapon
    for(int i = 0; i < n; i++)
        cin >> c[i];
 
    vector<vector<int>> a(n, vector<int>(n));    // damage of attacks
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            cin >> a[i][j];
        }
    }

    // goal is to open all boxes with min attacks
    vector<int> vis(n, 0);   // For marking weapons with 0 durability
    vector<pair<int,int>> results;
    
    for(int i = 0; i < n; i++)
    {
        // For every box
        while(h[i] > 0)
        {
            // Choose the max reduction weapon that's still available
            int max_val = -1;
            int weapon = -1;
            
            for(int j = 0; j < n; j++)  // FIXED: was j < n; i++
            {
                if(!vis[j] && a[j][i] > max_val)
                {
                    max_val = a[j][i];
                    weapon = j;
                }
            }
            
            if(weapon == -1)
            {
                // No weapon available, use bare hands
                results.push_back({-1, i});
                h[i]--;
            }
            else
            {
                // Calculate how many attacks needed
                int attacks_needed = (h[i] + max_val - 1) / max_val;  // Ceiling division
                int attacks_to_do = min(attacks_needed, c[weapon]);
                
                // Perform the attacks
                for(int k = 0; k < attacks_to_do; k++)
                {
                    results.push_back({weapon, i});
                }
                
                // Update durability and hardness
                c[weapon] -= attacks_to_do;
                h[i] -= max_val * attacks_to_do;
                
                // Mark weapon as used up if durability is 0
                if(c[weapon] == 0)
                {
                    vis[weapon] = 1;
                }
            }
        }
    }

    // Output results
    for(auto it : results)
    {
        cout << it.first << " " << it.second << "\n";
    }

    return 0;
}