#include <bits/stdc++.h>
using namespace std;

int main() {

    int n;
    cin >> n;

    cout << n << " " << n*(n-1)/2 << endl;
    for(int i=0; i<n-1; i++)
        for(int j=i+1; j<n; j++)
            cout << i << " " << j << endl;
}