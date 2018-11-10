#include <bits/stdc++.h>
#include <cuda.h>
#include "Graph.h"

#define CEIL(a, b) ((a - 1) / b + 1)

using namespace std;

__device__ bool *device_conflictExists;
__device__ bool *device_conflicts;
__device__ bool *device_colours;
__device__ int nodeCount;



// need: graph, device_conflicts
__global__ void assignColoursKernel(Graph *graph)
{

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= graph->getNodeCount() || !device_conflicts[node])
        return;

    int maxColours = graph->getMaxDegree() + 1;
    // Create forbidden array of size maxDegree
    bool *forbidden = new bool[maxColours + 1];
    memset(forbidden, false, sizeof(bool) * (maxColours + 1));

    vector<int> neighbours = graph->getAdjacencyList(node);
    for (int neighbour : neighbours)
        forbidden[device_colours[neighbour]] = true;

    // __syncthreads();
    for (int colour = 1; colour <= maxColours; ++colour)
    {
        if (forbidden[colour] == false)
        {
            // TODO: Check if needs to be synced
            device_colours[node] = colour;
            break;
        }
    }
    *device_conflictExists = false;
}

void assignColours(Graph *graph)
{

    // Launch assignColoursKernel with nodeCount number of threads
    assignColoursKernel<<<CEIL(nodeCount, 1024), 1024>>>(graph);
}

__global__ void detectConflictsKernel(Graph *graph)
{

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= graph->getNodeCount())
        return;
    *device_conflictExists = false;
    device_conflicts[node] = false;
    vector<int> neighbours = graph->getAdjacencyList(node);
    for (int neighbour : neighbours)
    {
        if (device_colours[neighbour] == device_colours[node] && neighbour < node)
        {
            //conflict
            device_conflicts[node] = true;
            *device_conflictExists = true;
        }
    }
}

bool detectConflicts(Graph *graph)
{

    //Launch detectConflictsKernel with nodeCount number of threads
    detectConflictsKernel<<<CEIL(nodeCount, 1024), 1024>>>(graph);

    // Copy device_conflictExists to conflictExists and return
    bool conflictExists;
    cudaMemcpy(device_conflictExists, (void *)&conflictExists, sizeof(bool), cudaMemcpyDeviceToHost);

    return conflictExists;
}

int *graphColouring(Graph *graph)
{

    // Initialize a boolean array of size = number of nodes for set of nodes with conflicts.
    // vector<bool> conflicts(graph->getNodeCount(), true);

    // Initialize all nodes to invalid colour (0)
    // vector<int> colours(graph->getNodeCount(), 0);
    // bool conflictExists = true;

    do
    {
        assignColours(graph);
    } while (detectConflicts(graph));

    // Copy colours to host and return
    int *host_colours = new int[nodeCount];
    cudaMemcpy(host_colours, device_colours, sizeof(device_colours), cudaMemcpyDeviceToHost);

    return host_colours;
}

int main()
{

    Graph *h_graph = new Graph();
    Graph *d_graph;

    cudaMalloc((void **)&d_graph, sizeof(Graph));
    h_graph->readGraph();
    nodeCount = h_graph->getNodeCount();
    cudaMemcpy(d_graph, h_graph, sizeof(Graph), cudaMemcpyHostToDevice);

    bool *host_conflicts = new bool[nodeCount];
    memset(host_conflicts, false, sizeof(bool) * nodeCount);

    cudaMalloc((void**)&device_conflictExists, sizeof(bool));
    cudaMalloc((void**)&device_conflicts, sizeof(bool));
    cudaMalloc((void**)&device_colours, sizeof(int) * nodeCount);

    cudaMemcpy(device_conflicts, host_conflicts, sizeof(host_conflicts), cudaMemcpyHostToDevice);

    // Free all memory
    return 0;
}