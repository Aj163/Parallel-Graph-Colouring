#include <iostream>
#include "Graph.h"

using namespace std;

void assignColours(Graph *graph, int nodeCount,
                   int *colours, bool *conflicts, int maxDegree)
{

    int maxColours = maxDegree + 1;

    for (int node = 0; node < nodeCount; ++node)
    {
        if (!conflicts[node])
            break;

        bool *forbidden = new bool[maxColours + 1];
        memset(forbidden, false, sizeof(bool) * (maxColours + 1));

        for (int i = graph->adjacencyListPointers[node]; i < graph->adjacencyListPointers[node + 1]; i++)
        {
            int neighbour = graph->adjacencyList[i];
            forbidden[colours[neighbour]] = true;
        }

        for (int colour = 1; colour <= maxColours; ++colour)
        {
            if (forbidden[colour] == false)
            {
                colours[node] = colour;
                break;
            }
        }

        delete[] forbidden;
    }
}

bool detectConflicts(Graph *graph, int nodeCount, int *colours, bool *conflicts)
{

    bool conflictExists = false;

    for (int node = 0; node < nodeCount; ++node)
    {
        if (node >= nodeCount)
            break;

        conflicts[node] = false;

        for (int i = graph->adjacencyListPointers[node]; i < graph->adjacencyListPointers[node + 1]; i++)
        {
            int neighbour = graph->adjacencyList[i];

            if (colours[neighbour] == colours[node] && neighbour < node)
            {
                conflicts[node] = true;
                conflictExists = true;
            }
        }
    }

    return conflictExists;
}

int *graphColouring(Graph *graph, int nodeCount, int maxDegree)
{

    // Boolean array for conflicts
    bool *conflicts = new bool[nodeCount];
    int *colours = new int[nodeCount];

    // Initialize all nodes to invalid colour (0)
    memset(colours, 0, sizeof(int) * nodeCount);
    // Initialize all nodes into conflict
    memset(conflicts, true, sizeof(bool) * nodeCount);

    do
    {
        assignColours(graph, nodeCount, colours, conflicts, maxDegree);

    } while (detectConflicts(graph, nodeCount, colours, conflicts));

    delete[] conflicts;

    return colours;
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

    Graph *graph = new Graph();
    graph->readGraph();

    int nodeCount = graph->getNodeCount();
    int edgeCount = graph->getEdgeCount();
    int maxDegree = graph->getMaxDegree();
    
    clock_t start, end;
    start = clock();

    int *colouring = graphColouring(graph, nodeCount, maxDegree);

    end = clock();
    float time_taken = 1000.0* (end - start)/CLOCKS_PER_SEC;

    int totalColours = INT_MIN;
    for (int i = 0; i < nodeCount; i++)
    {
        totalColours = max(totalColours, colouring[i]);
        if(choice == 'y' || choice == 'Y')
            printf("Node %d => Colour %d\n", i, colouring[i]);
    }
    cout << endl;
    printf("\nNumber of colours required (chromatic number) ==> %d\n", totalColours);
    printf("Time Taken (Serial) = %f ms\n", time_taken);

    if (argc == 3)
    {
        freopen(argv[2], "w", stdout);
        for (int i = 0; i < nodeCount; i++)
            cout << colouring[i] << " ";
        cout << endl;
    }

    // Free all memory
    delete[] colouring;
    delete graph;
}
