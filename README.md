## Parallel-Graph-Colouring
CUDA C/C++ implementation for [Parallel Graph Coloring Heuristics for Manycore Architectures](https://ieeexplore.ieee.org/abstract/document/7516086)

### Graph Colouring
Graph coloring is a simple way of labelling graph components such as vertices, edges, or regions under some constraints. 

A **vertex coloring** is a type of Graph colouring problem which finds its application in many areas.
Vertex coloring is an assignment of colors to the vertices of a graph such that no two adjacent vertices have the same color, i.e. no two vertices of an edge should be of the same color.

A vertex coloring that minimize the number of colors needed for a given graph G is known as a minimum vertex coloring of G.
The minimum number of colors itself is called the chromatic number, and a graph with chromatic number = k is said to be a k-chromatic graph.

Unfortunately, there is no efficient algorithm available for coloring a graph with minimum number of colors as the problem is a known NP Complete problem.
There are some algorithms like Brelaz's heuristic algorithm which provides a good approximation for the minimum vertex colouring problem.