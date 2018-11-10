#include <bits/stdc++.h>
#include <cuda.h>
#include "Graph.h"

#define CEIL(a, b) ((a - 1) / b + 1)

using namespace std;


__global__ void assignColoursKernel(Graph *graph, int nodeCount, int edgeCount, 
    int *device_colours, bool *device_conflicts, int maxDegree) {

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodeCount || !device_conflicts[node])
        return;

    int maxColours = maxDegree + 1;
    // Create forbidden array of size maxDegree
    bool *forbidden = new bool[maxColours + 1];
    memset(forbidden, false, sizeof(bool) * (maxColours + 1));

    for (int i=graph->adjacencyListPointers[node]; i<graph->adjacencyListPointers[node +1]; i++) {
        int neighbour = graph->adjacencyList[i];
        forbidden[device_colours[neighbour]] = true;
    }

    // __syncthreads();
    for (int colour = 1; colour <= maxColours; ++colour) {
        if (forbidden[colour] == false) {
            // TODO: Check if needs to be synced
            device_colours[node] = colour;
            break;
        }
    }
}

void assignColours(Graph *graph, int nodeCount, int edgeCount, 
    int *device_colours, bool *device_conflicts, int maxDegree) {

    // Launch assignColoursKernel with nodeCount number of threads
    assignColoursKernel <<< CEIL(nodeCount, 1024), 1024 >>> 
        (graph, nodeCount, edgeCount, device_colours, device_conflicts, maxDegree);
}

__global__ void detectConflictsKernel(Graph *graph, int nodeCount, int edgeCount, 
    int *device_colours, bool *device_conflicts, bool *device_conflictExists) {

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodeCount)
        return;

    device_conflicts[node] = false;
    
    for (int i=graph->adjacencyListPointers[node]; i<graph->adjacencyListPointers[node +1]; i++) {
        int neighbour = graph->adjacencyList[i];
        if (device_colours[neighbour] == device_colours[node] && neighbour < node) {
            //conflict
            device_conflicts[node] = true;
            *device_conflictExists = true;
        }
    }
}

bool detectConflicts(Graph *graph, int nodeCount, int edgeCount, 
    int *device_colours, bool *device_conflicts) {

    bool *conflictExists = new bool;
    bool *device_conflictExists;

    *conflictExists = false;
    cudaMalloc((void**)&device_conflictExists, sizeof(bool));
    cudaMemcpy(device_conflictExists, conflictExists, sizeof(bool), cudaMemcpyHostToDevice);

    //Launch detectConflictsKernel with nodeCount number of threads
    detectConflictsKernel <<< CEIL(nodeCount, 1024), 1024 >>> 
        (graph, nodeCount, edgeCount, device_colours, device_conflicts, device_conflictExists);

    // Copy device_conflictExists to conflictExists and return
    cudaMemcpy(device_conflictExists, conflictExists, sizeof(bool), cudaMemcpyDeviceToHost);

    return conflictExists;
}

int *graphColouring(Graph *graph, int nodeCount, int edgeCount, int maxDegree) {

    // Boolean array for conflicts
    bool * host_conflicts = new bool[nodeCount];
    int *host_colours = new int[nodeCount];
    int *device_colours;
    bool *device_conflicts;

    // Initialize all nodes to invalid colour (0)
    memset(host_colours, 0, sizeof(int) * nodeCount);
    // Initialize all nodes into conflict
    memset(host_conflicts, 1, sizeof(int) * nodeCount);

    cudaMalloc((void**)&device_colours, sizeof(int) * nodeCount);
    cudaMemcpy(device_colours, host_colours, sizeof(int) * nodeCount, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&device_conflicts, sizeof(bool) * nodeCount);
    cudaMemcpy(device_conflicts, host_conflicts, sizeof(int) * nodeCount, cudaMemcpyHostToDevice);

    do {
        assignColours(graph, nodeCount, edgeCount, device_colours, device_conflicts, maxDegree);
    } while (detectConflicts(graph, nodeCount, edgeCount, device_colours, device_conflicts));

    // Copy colours to host and return
    cudaMemcpy(host_colours, device_colours, sizeof(int) * nodeCount, cudaMemcpyDeviceToHost);

    return host_colours;
}

int main() {

    Graph *h_graph = new Graph();
    Graph *d_graph;

    cudaMalloc((void **)&d_graph, sizeof(Graph));
    h_graph->readGraph();

    int nodeCount = h_graph->getNodeCount();
    int edgeCount = h_graph->getEdgeCount();
    int maxDegree = h_graph->getMaxDegree();
    cudaMemcpy(d_graph, h_graph, sizeof(Graph), cudaMemcpyHostToDevice);

    // Copy Adjancency List to device
    int *adjacencyList;
    // Alocate device memory and copy 
    cudaMalloc((void**)&adjacencyList, sizeof(int) * (2 * edgeCount +1));
    cudaMemcpy(adjacencyList, h_graph->adjacencyList, sizeof(int) * (2 * edgeCount +1), cudaMemcpyHostToDevice);
    // Update the pointer to this, in d_graph
    cudaMemcpy(&(d_graph->adjacencyList), &adjacencyList, sizeof(int*), cudaMemcpyHostToDevice);

    // Copy Adjancency List Pointers to device
    int *adjacencyListPointers;
    // Alocate device memory and copy 
    cudaMalloc((void**)&adjacencyListPointers, sizeof(int) * (nodeCount +1));
    cudaMemcpy(adjacencyListPointers, h_graph->adjacencyListPointers, sizeof(int) * (nodeCount +1), cudaMemcpyHostToDevice);
    // Update the pointer to this, in d_graph
    cudaMemcpy(&(d_graph->adjacencyListPointers), &adjacencyListPointers, sizeof(int*), cudaMemcpyHostToDevice);

    cout << "Hi\n";
    int *colouring = graphColouring(d_graph, nodeCount, edgeCount, maxDegree);

    for(int i=0; i<nodeCount; i++)
        cout << colouring[i] << " ";
    cout << endl;

    // Free all memory
    delete[] colouring;
}