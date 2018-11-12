#include <bits/stdc++.h>
using namespace std;

int main(int argc, char *argv[]) {

    if(argc != 3) {
        cout << "Try ./a.out <graph> <colouring>\n";
        return 0;
    }

    int n, m, u, v, C=0;
    ifstream graph(argv[1]);
    ifstream colour(argv[2]);

    graph >> n >> m;
    vector<int> *g = new vector<int>[n];
    
    for(int i=0; i<n; i++) {
        graph >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }

    int *colours = new int[n];
    for(int i=0; i<n; i++) {
        colour >> colours[i];
        if(colours[i] <= 0) {
            cout << "WA\n";
            return 0;
        }
        C = max(C, colours[i]);
    }

    colour.close();
    graph.close();

    bool AC = 1;
    for(int i=0; i<n; i++)
        for(int node : g[i])
        if(colours[i] == colours[node]) {
            AC = 0;
            break;
        }

    if(AC == 0) {
        cout << "WA\n";
        return 0;
    }

    delete[] g, colours;
    cout << "AC\nNumber of colours used : " << C << endl;
}