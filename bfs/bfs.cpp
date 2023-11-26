#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
// void top_down_step(
//     Graph g,
//     vertex_set* frontier,
//     vertex_set* new_frontier,
//     int* distances)
// {

//     // going through all the nodes on the frontier
//     for (int i=0; i<frontier->count; i++) {

//         int node = frontier->vertices[i];

//         // get the start and end for the chunk of neighbors for this node
//         int start_edge = g->outgoing_starts[node];
//         int end_edge = (node == g->num_nodes - 1)
//                            ? g->num_edges
//                            : g->outgoing_starts[node + 1];

//         // attempt to add all neighbors to the new frontier
//         for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
//             int outgoing = g->outgoing_edges[neighbor];
//             // we do not wish to add any neighbors that already exists in our frontier 
//             if (distances[outgoing] == NOT_VISITED_MARKER) {
//                 distances[outgoing] = distances[node] + 1;
//                 int index = new_frontier->count++;
//                 new_frontier->vertices[index] = outgoing;
//             }
//         }
//     }
// }
 


// BEGINNING OF NEW TOP_DOWN_STEP
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{

    // going through all the nodes on the frontier
    // first we notice that each node can add to the frontier independently

    #pragma omp parallel for 
    for (int i=0; i<frontier->count; i++) {
        int node = frontier->vertices[i];

        // get the start and end for the chunk of neighbors for this node
        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];
        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];
            // we do not wish to add any neighbors that already exist in our frontier 
            if (distances[outgoing] == NOT_VISITED_MARKER) {
                distances[outgoing] = distances[node] + 1;

                int index = new_frontier->count;
                while (!__sync_bool_compare_and_swap(&new_frontier->count, index, new_frontier->count+1)) {
                    index = new_frontier->count;
                }
                new_frontier->vertices[index] = outgoing;
            }
        }
    }
}
// END OF NEW TOP_DOWN_STEP



// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bottom_up_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    for (Vertex v = 0; v < g->num_nodes; v++) {
        if (distances[v] != NOT_VISITED_MARKER)
            continue;
        Vertex start_edge = g->incoming_starts[v];
        Vertex end_edge = (v == g->num_nodes - 1)
                        ? g->num_edges
                        : g->incoming_starts[v + 1];
        bool added = false;
        for (int u_id = start_edge; u_id < end_edge; u_id++) {
            Vertex u = g->incoming_edges[u_id];
            for (int i = 0; i < frontier->count; i++) {
                if (frontier->vertices[i] != u)
                    continue;
                added = true;
                int index = new_frontier->count++;
                new_frontier->vertices[index] = v;
                distances[v] = distances[u] + 1;
                break;
            }
            if (added) {
                break;
            }
        }
    }
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    // print_graph(graph);
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    for(int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }
    
    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {
        vertex_set_clear(new_frontier);
        bottom_up_step(graph, frontier, new_frontier, sol->distances);
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
