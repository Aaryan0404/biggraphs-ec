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
#define TOP_DOWN 1
#define BOTTOM_UP 2

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
 

void top_down_step_fast(Graph g,
    vertex_set* frontier,
    vertex_set** v_sets,
    int* distances) 
{
    int n_threads = omp_get_max_threads();
    vertex_set* vertex_sets = *v_sets;
    int chunk = (int) (frontier->count / (n_threads * 100)) + 100;
    #pragma omp parallel for schedule(dynamic, chunk)
    for (int i = 0; i < frontier->count; i++) {
        int tid = omp_get_thread_num();
        int node = frontier->vertices[i];
        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                        ? g->num_edges
                        : g->outgoing_starts[node + 1];
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];
            // if some other thread is taking care of this node, then this thread ignores
            if (!__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1))
                continue;
            // trivially atomic because it is a per-thread count
            int index = vertex_sets[tid].count++;
            vertex_sets[tid].vertices[index] = outgoing;
        }
    }
    // in this variant, we don't maintain tmp `new_frontier` buffer; we directly update old frontier
    // this is because synchronizing threads is too much overhead
    vertex_set_clear(frontier);
    // we want to find unique spots for the threads to place their discovered nodes; hence we accum
    int* cum = new int[n_threads];
    for (int i = 0; i < n_threads+1; i++) {
        if (i == 0) {
            cum[i] = 0;
        } else {
            cum[i] = vertex_sets[i-1].count + cum[i-1];
        }
    }

    // this is where we actually position the nodes in their spots
    #pragma omp parallel for
    for (int i = 0; i < n_threads; i++) {
        vertex_set v = vertex_sets[i];
        for (int j = 0; j < v.count; j++) {
            frontier->vertices[cum[i] + j] = v.vertices[j];
        }
        // reset for next top_down_step
        vertex_sets[i].count = 0;
    }

    // our frontier count is just the last entry in cum
    frontier->count = cum[n_threads];
    delete cum;
}


// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    double start, end;
    vertex_set list1;
    vertex_set_init(&list1, graph->num_nodes);

    vertex_set* frontier = &list1;

    // initialize all nodes to NOT_VISITED
    int chunk = (int) (graph->num_nodes / omp_get_max_threads() * 100) + 100;
    #pragma omp parallel for schedule(dynamic, chunk)
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;
    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
    
    int n_threads = omp_get_max_threads();
    vertex_set* v_sets = new vertex_set[n_threads];
    #pragma omp parallel for schedule(dynamic, chunk)
    for (int i = 0; i < n_threads; i++) {
        vertex_set_init(&v_sets[i], graph->num_nodes);
    }
    int i = 0;
    while (frontier->count != 0) {
        i++;
        top_down_step_fast(graph, frontier, &v_sets, sol->distances);
    }
    delete v_sets;
}



void bottom_up_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{
    return;
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

int bottom_up_step_fast(
    Graph g,
    int* frontier,
    int* distances,
    int max_distance)
{
    // every BFS step pushes out frontier by 1 (i.e. adding 1 to the max distance found in graph)
    int next_distance = max_distance + 1;
    int amnt_done = 0;
    int num_threads = omp_get_max_threads();
    int chunk = (int) (g->num_nodes / (num_threads * 100)) + 100;

    #pragma omp parallel for
    for (Vertex v=0; v<g->num_nodes; v++) {
        if (distances[v] != NOT_VISITED_MARKER)
            continue;
        Vertex start_edge = g->incoming_starts[v];
        Vertex end_edge = (v == g->num_nodes - 1)
                        ? g->num_edges
                        : g->incoming_starts[v+1];
        for (int u_idx = start_edge; u_idx < end_edge; u_idx++) {
            Vertex u = g->incoming_edges[u_idx];
            // check if this neighbor is a part of the latest frontier
            if (frontier[u] == max_distance) {
                distances[v] = distances[u] + 1;
                frontier[v] = next_distance;
                #pragma atomic
                amnt_done++;
                break; // early exit bc we just need at LEAST one neighbor in the latest frontier
            }
        }
    } 
    return amnt_done;
}


void bfs_bottom_up(Graph graph, solution* sol)
{
    int* frontier = new int[graph->num_nodes];
    #pragma omp parallel for
    for(int i=0; i<graph->num_nodes; i++) {
        frontier[i] = sol->distances[i] = NOT_VISITED_MARKER;
    }
    
    // setup root node
    frontier[ROOT_NODE_ID] = sol->distances[ROOT_NODE_ID] = 0;
    int distances_count = 1; // we init with the trivial distance of the root node to itself
    int max_distance = 0;
    while (distances_count != 0) {
        distances_count = bottom_up_step_fast(graph, frontier, sol->distances, max_distance++);
    }
    delete frontier;
}


void bottom_to_top(
    Graph g, vertex_set* top_frontier, 
    int* bottom_frontier, int max_distance, 
    vertex_set* v_sets) 
{
    int n_threads = omp_get_max_threads();
    // clear out the vertex sets (might have results from a old top down step)
    for (int i = 0; i < n_threads; i++) {
        vertex_set_clear(&v_sets[i]);
    }
    #pragma omp parallel for
    for (int i = 0; i < g->num_nodes; i++) {
        int tid = omp_get_thread_num();
        vertex_set* vset = &v_sets[tid];
        if (bottom_frontier[i] == max_distance) {
            // trivially atomic because of indexing vsets by tid
            vset->vertices[vset->count++] = i;
        }
    }
    top_frontier->count = 0;

}

void top_to_bottom(Graph g, vertex_set* top_frontier, int* bottom_frontier, int& max_distance, vertex_set* v_sets) {
    int n_threads = omp_get_max_threads();
    max_distance++;
    #pragma omp parallel for
    for (int i = 0; i < top_frontier->count; i++) {
        bottom_frontier[top_frontier->vertices[i]] = max_distance;
    }
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // since we are leveraging both the bottom up and top down implementations from previous parts
    // we must init the data structures used in those implementations
    int* bottom_frontier = new int[graph->num_nodes];
    vertex_set list1, list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set* frontier = &list1;
    int n_threads = omp_get_max_threads();
    // setup top down frontier (this is basically the same from top_down() above)
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;

    vertex_set* v_sets = new vertex_set[n_threads];
    #pragma omp parallel
    {
    #pragma omp for nowait
    for (int i = 0; i < n_threads; i++) {
        vertex_set_init(&v_sets[i], graph->num_nodes);
    }
    #pragma omp for
    for (int i = 0; i < graph->num_nodes; i++) {
        bottom_frontier[i] = sol->distances[i] = NOT_VISITED_MARKER;
    }
    }

    bottom_frontier[0] = sol->distances[ROOT_NODE_ID] = 0;
    int max_distance = 0;
    bool prev_state = 0;
    bool do_top_down = true;
    // the following two variables will provide heuristic of when to txn btwn top-down & bottom-up
    int n_nodes_to_explore = graph->num_nodes;
    int frontier_size = 0;

    while (true) {
        // if there is at least x times more nodes in the frontier set than in the unexplored,
        // we def want to do bottom-up because bottom up allows for unexplored nodes to look up to the set
        if (do_top_down && (frontier_size * 100) / n_nodes_to_explore > 10) {
            do_top_down = false;
        }

        if (do_top_down) {
            // we must convert our representations to match the variant that will be used in this step
            if (prev_state == BOTTOM_UP)
                bottom_to_top(graph, frontier, bottom_frontier, max_distance, v_sets);
            top_down_step_fast(graph, frontier, &v_sets, sol->distances);
            // same exit condition as in above top_down()
            if (frontier->count == 0)
                break;
            prev_state = TOP_DOWN;
            n_nodes_to_explore -= frontier->count;
        } else {
            if (prev_state == TOP_DOWN)
                top_to_bottom(graph, frontier, bottom_frontier, max_distance, v_sets);
            int amnt_added = bottom_up_step_fast(graph, bottom_frontier, sol->distances, max_distance++);
            if (amnt_added == 0)
                break;
            prev_state = BOTTOM_UP;
            n_nodes_to_explore -= amnt_added;
        }
    }
}