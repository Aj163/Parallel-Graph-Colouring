#include <bits/stdc++.h>
using namespace std;

class Graph {

private:
	int nodeCount, edgeCount;
	int maxDegree;
	vector<int> *adjacencyList;

public:
	// Graph(int n)
	
	// {
	// 	this->nodeCount = n;
	// 	this->adjacencyList = new vector<int>(n);
	// }

	__host__ __device__ int getNodeCount() {
		return nodeCount;
	}

	__host__ __device__ int getMaxDegree() {
		return maxDegree;
	}

	void readGraph() {

		int u, v;
		cin >> nodeCount >> edgeCount;
		adjacencyList = new vector<int>[nodeCount];
		for (int i = 0; i < edgeCount; i++) {
			cin >> u >> v;
			adjacencyList[u].push_back(v);
			adjacencyList[v].push_back(u);
		}

		maxDegree = INT_MAX;
		for(int i=0; i<nodeCount; i++)
			maxDegree = max(maxDegree, (int)adjacencyList[i].size());
	}

	__host__ __device__ vector<int> getAdjacencyList(int node) {
		// if(node < nodeCount)
			return adjacencyList[node];
		// return NULL;
	}

	__host__ __device__ ~Graph() {
		delete[] adjacencyList;
	}
};