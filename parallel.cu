/**
*   CUDA C/C++ implementation for Parallel Graph Coloring for Manycore Architectures
*   {@link https://ieeexplore.ieee.org/abstract/document/7516086}
*
*   @author Ashwin Joisa
*   @author Praveen Gupta
**/

//=============================================================================================//

// Include header files
#include <iostream>
#include <cuda.h>

// Include custom header file for implementation of Graphs
#include "Graph.h"

//=============================================================================================//

#define MAX_THREAD_COUNT 1024
#define CEIL(a, b) ((a - 1) / b + 1)

//=============================================================================================//

using namespace std;

//=============================================================================================//

// Catch Cuda errors
void catchCudaError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        printf("\n====== Cuda Error Code %i ======\n %s\n", error, cudaGetErrorString(error));
        exit(-1);
    }
}
//=============================================================================================//

__global__ void assignColoursKernel(Graph *graph, int nodeCount,
                                    int *device_colours, bool *device_conflicts, int maxDegree)
{

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodeCount || !device_conflicts[node])
        return;

    int maxColours = maxDegree + 1;
    // Create forbidden array of size maxDegree
    bool *forbidden = new bool[maxColours + 1];
    memset(forbidden, false, sizeof(bool) * (maxColours + 1));

    for (int i = graph->adjacencyListPointers[node]; i < graph->adjacencyListPointers[node + 1]; i++)
    {
        int neighbour = graph->adjacencyList[i];
        forbidden[device_colours[neighbour]] = true;
    }

    for (int colour = 1; colour <= maxColours; ++colour)
    {
        if (forbidden[colour] == false)
        {
            device_colours[node] = colour;
            break;
        }
    }

    delete[] forbidden;
}

void assignColours(Graph *graph, int nodeCount,
                   int *device_colours, bool *device_conflicts, int maxDegree)
{

    // Launch assignColoursKernel with nodeCount number of threads
    assignColoursKernel<<<CEIL(nodeCount, MAX_THREAD_COUNT), MAX_THREAD_COUNT>>>(graph, nodeCount, device_colours, device_conflicts, maxDegree);
}

__global__ void detectConflictsKernel(Graph *graph, int nodeCount,
                                      int *device_colours, bool *device_conflicts, bool *device_conflictExists)
{

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodeCount)
        return;

    device_conflicts[node] = false;

    for (int i = graph->adjacencyListPointers[node]; i < graph->adjacencyListPointers[node + 1]; i++)
    {
        int neighbour = graph->adjacencyList[i];
        if (device_colours[neighbour] == device_colours[node] && neighbour < node)
        {
            //conflict
            device_conflicts[node] = true;
            *device_conflictExists = true;
        }
    }
}

bool detectConflicts(Graph *graph, int nodeCount,
                     int *device_colours, bool *device_conflicts)
{

    bool *device_conflictExists;
    bool conflictExists = false;

    cudaMalloc((void **)&device_conflictExists, sizeof(bool));
    cudaMemcpy(device_conflictExists, &conflictExists, sizeof(bool), cudaMemcpyHostToDevice);

    //Launch detectConflictsKernel with nodeCount number of threads
    detectConflictsKernel<<<CEIL(nodeCount, MAX_THREAD_COUNT), MAX_THREAD_COUNT>>>(graph, nodeCount, device_colours, device_conflicts, device_conflictExists);

    // Copy device_conflictExists to conflictExists and return
    cudaMemcpy(&conflictExists, device_conflictExists, sizeof(bool), cudaMemcpyDeviceToHost);
    
    // Free device memory
    catchCudaError(cudaFree(device_conflictExists));
    
    return conflictExists;
}

int *graphColouring(Graph *graph, int nodeCount, int maxDegree)
{

    // Boolean array for conflicts
    bool *host_conflicts = new bool[nodeCount];
    int *host_colours = new int[nodeCount];
    int *device_colours;
    bool *device_conflicts;

    // Initialize all nodes to invalid colour (0)
    memset(host_colours, 0, sizeof(int) * nodeCount);
    // Initialize all nodes into conflict
    memset(host_conflicts, true, sizeof(bool) * nodeCount);

    cudaMalloc((void **)&device_colours, sizeof(int) * nodeCount);
    cudaMemcpy(device_colours, host_colours, sizeof(int) * nodeCount, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&device_conflicts, sizeof(bool) * nodeCount);
    cudaMemcpy(device_conflicts, host_conflicts, sizeof(bool) * nodeCount, cudaMemcpyHostToDevice);

    do
    {
        assignColours(graph, nodeCount, device_colours, device_conflicts, maxDegree);
    } while (detectConflicts(graph, nodeCount, device_colours, device_conflicts));

    // Copy colours to host and return
    cudaMemcpy(host_colours, device_colours, sizeof(int) * nodeCount, cudaMemcpyDeviceToHost);

    delete[] host_conflicts;
    catchCudaError(cudaFree(device_colours));
    catchCudaError(cudaFree(device_conflicts));

    return host_colours;
}

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " <graph_input_file> [output_file]\n";
        return 0;
    }

    char choice;
    cout << "Would you like to print the colouring of the graph? (y/n) ";
    cin >> choice;

    freopen(argv[1], "r", stdin);

    Graph *host_graph = new Graph();
    Graph *device_graph;

    cudaMalloc((void **)&device_graph, sizeof(Graph));
    host_graph->readGraph();

    int nodeCount = host_graph->getNodeCount();
    int edgeCount = host_graph->getEdgeCount();
    int maxDegree = host_graph->getMaxDegree();
    cudaMemcpy(device_graph, host_graph, sizeof(Graph), cudaMemcpyHostToDevice);

    // Copy Adjancency List to device
    int *adjacencyList;
    // Alocate device memory and copy
    cudaMalloc((void **)&adjacencyList, sizeof(int) * (2 * edgeCount + 1));
    cudaMemcpy(adjacencyList, host_graph->adjacencyList, sizeof(int) * (2 * edgeCount + 1), cudaMemcpyHostToDevice);
    // Update the pointer to this, in device_graph
    cudaMemcpy(&(device_graph->adjacencyList), &adjacencyList, sizeof(int *), cudaMemcpyHostToDevice);

    // Copy Adjancency List Pointers to device
    int *adjacencyListPointers;
    // Alocate device memory and copy
    cudaMalloc((void **)&adjacencyListPointers, sizeof(int) * (nodeCount + 1));
    cudaMemcpy(adjacencyListPointers, host_graph->adjacencyListPointers, sizeof(int) * (nodeCount + 1), cudaMemcpyHostToDevice);
    // Update the pointer to this, in device_graph
    cudaMemcpy(&(device_graph->adjacencyListPointers), &adjacencyListPointers, sizeof(int *), cudaMemcpyHostToDevice);

    cudaEvent_t device_start, device_end;
    catchCudaError(cudaEventCreate(&device_start));
    catchCudaError(cudaEventCreate(&device_end));
    catchCudaError(cudaEventRecord(device_start));

    int *colouring = graphColouring(device_graph, nodeCount, maxDegree);

    catchCudaError(cudaEventRecord(device_end));
    
    float device_time_taken;

    cudaEventSynchronize(device_end);
    cudaEventElapsedTime(&device_time_taken, device_start, device_end);

    int chromaticNumber = INT_MIN;
    for (int i = 0; i < nodeCount; i++)
    {
        chromaticNumber = max(chromaticNumber, colouring[i]);
        if(choice == 'y' || choice == 'Y')
            printf("Node %d => Colour %d\n", i, colouring[i]);
    }
    cout << endl;
    printf("\nNumber of colours used (chromatic number) ==> %d\n", chromaticNumber);
    printf("Time Taken (Parallel) = %f ms\n", device_time_taken);

    if (argc == 3)
    {
        freopen(argv[2], "w", stdout);
        for (int i = 0; i < nodeCount; i++)
            cout << colouring[i] << " ";
        cout << endl;
    }

    // Free all memory
    delete[] colouring;
    catchCudaError(cudaFree(adjacencyList));
    catchCudaError(cudaFree(adjacencyListPointers));
    catchCudaError(cudaFree(device_graph));
}