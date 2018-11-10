#include <bits/stdc++.h>
using namespace std;

int main()
{
    int n, m;
    int x, y;

    cin >> n >> m;
    if (m < n - 1 || m > n * 1ll * (n - 1) / 2) {
        // Can't be a simple connected graph
        return 0;
    }

    cout << n << " " << m << endl;

    int nodes[n + 1];
    pair<int, int> edges[m + 1];
    map<pair<int, int>, bool> edgeExists;

    for (int i = 0; i < n; i++) {
        nodes[i] = i;
    }

    srand(time(0));
    random_shuffle(nodes, nodes + n);
    for (int i = 1; i < n; i++) {
        int v = rand() % i;
        x = min(nodes[i], nodes[v]);
        y = max(nodes[i], nodes[v]);
        edges[i - 1] = make_pair(x, y);
        edgeExists[make_pair(x, y)] = 1;
    }

    for (int i = n - 1; i < m; i++) {
        do {
            x = rand() % n;
            y = rand() % n;

            if (x > y)
                swap(x, y);
        } while (x == y || edgeExists[make_pair(x, y)]);

        edges[i] = make_pair(x, y);
        edgeExists[make_pair(x, y)] = 1;
    }

    random_shuffle(edges, edges + m);
    for (int i = 0; i < m; i++) {
        cout << edges[i].first << " " << edges[i].second << endl;
    }
}
