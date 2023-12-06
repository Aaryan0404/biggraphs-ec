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
 
void top_down_step_fast(Graph g,
    vertex_set* frontier,
    vertex_set** v_sets,
    int* distances) 
{
    int n_threads = omp_get_max_threads();
    vertex_set* vertex_sets = *v_sets;
    int node_section_size = 100 + ((int) (frontier->count / (n_threads * 100)))/(n_threads/8);

    #pragma omp parallel for schedule(dynamic, node_section_size)
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
            int old = distances[outgoing];
            if (old == NOT_VISITED_MARKER) {
                if (!__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1))
                    continue;
                int index = vertex_sets[tid].count++;
                // trivially atomic because it is a per-thread count
                vertex_sets[tid].vertices[index] = outgoing;
            }
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
    int chunk = 100 + ((int) (graph->num_nodes / omp_get_max_threads() * 100))/(omp_get_max_threads()/8);

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
    while (frontier->count != 0) {
        top_down_step_fast(graph, frontier, &v_sets, sol->distances);
    }
    delete v_sets;
}

int bottom_up_step_fast(
    Graph g,
    unsigned char* frontier,
    int* distances,
    unsigned char clock)
{
    // every BFS step pushes out frontier by 1 (i.e. tick)
    unsigned char next_step_index = clock == 255 ? 1 : clock + 1;
    // int next_distance = max_distance + 1;
    int num_threads = omp_get_max_threads();
    int amnt_done[num_threads];
    memset(&amnt_done, 0, num_threads * sizeof(int));
    int node_section_size = 100 + ((int) (g->num_nodes / (num_threads * 100)) + 100)/(num_threads/8);

    #pragma omp parallel for schedule(dynamic, node_section_size)
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
            if (frontier[u] == clock) {
                distances[v] = distances[u] + 1;
                frontier[v] = next_step_index;
                amnt_done[omp_get_thread_num()]++;
                break; // early exit bc we just need at LEAST one neighbor in the latest frontier
            }
        }
    }
    int count = 0; 
    for (int i = 0; i < num_threads; i++) {
        count += amnt_done[i];
    }
    return count;
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    // unsigned char* frontier = new unsigned char[graph->num_nodes];
    unsigned char* frontier = (unsigned char*)malloc(sizeof(unsigned char) * graph->num_nodes);

    int num_threads = omp_get_max_threads(); 
    int node_section_size = 100 + ((int) (graph->num_nodes / (num_threads * 100)) + 100)/(num_threads/8);

    #pragma omp parallel for schedule(dynamic, node_section_size)
    for(int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
        frontier[i] = 0;
    }
    sol->distances[ROOT_NODE_ID] = 0;
    unsigned char clock = 1;
    frontier[ROOT_NODE_ID] = clock;
    int amnt_new_nodes = 1;
    int max_distance = 0;
    while (amnt_new_nodes != 0) {
        amnt_new_nodes = bottom_up_step_fast(graph, frontier, sol->distances, clock);
        clock = clock == 255 ? 1 : clock + 1;
    }
    delete frontier;
}


void bottom_to_top(
    Graph g, vertex_set* top_frontier, 
    unsigned char* bottom_frontier, unsigned char max_distance, 
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

void tick_clock(unsigned char* clock, unsigned char * frontier, int num_nodes) {
	(*clock)++;
	if (*clock == 255) {
		#pragma omp parallel for
		for (int i = 0; i < num_nodes; i++) {
			frontier[i] = frontier[i] == 255 ? 1 : 0;
		}
		(*clock) = 1;
	}
}

void top_to_bottom(Graph g, vertex_set* top_frontier, unsigned char* bottom_frontier, unsigned char* clock, vertex_set* v_sets) {
    int n_threads = omp_get_max_threads();
    tick_clock(clock, bottom_frontier, g->num_nodes);
    #pragma omp parallel for
    for (int i = 0; i < top_frontier->count; i++) {
        bottom_frontier[top_frontier->vertices[i]] = *clock;
    }
    
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // since we are leveraging both the bottom up and top down implementations from previous parts
    // we must init the data structures used in those implementations
    unsigned char* bottom_frontier = new unsigned char[graph->num_nodes];
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
    int node_section_size = 100 + ((int) (graph->num_nodes / (n_threads * 100)) + 100)/(n_threads/8);
    #pragma omp for schedule(dynamic, node_section_size)
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
        bottom_frontier[i] = 0;
    }
    }
    unsigned char clock = 1;
    sol->distances[ROOT_NODE_ID] = 0;
    bottom_frontier[ROOT_NODE_ID] = clock; 
    
    int prev_state = 0;
    bool do_top_down = true;
    // the following two variables will provide heuristic of when to txn btwn top-down & bottom-up
    int n_nodes_to_explore = graph->num_nodes;
    int frontier_size = 0;

    while (true) {
        // if there is at least x (x=15) times more nodes in the frontier set than in the unexplored,
        // we def want to do bottom-up because bottom up allows for unexplored nodes to look up to the set
        if (do_top_down && (frontier_size * 100) / n_nodes_to_explore > 15) {
            do_top_down = false;
        }
        if (do_top_down) {
            // we must convert our representations to match the variant that will be used in this step
            if (prev_state == BOTTOM_UP)
                bottom_to_top(graph, frontier, bottom_frontier, clock, v_sets);
            top_down_step_fast(graph, frontier, &v_sets, sol->distances);
            // same exit condition as in above top_down()
            if (frontier->count == 0)
                break;
            prev_state = TOP_DOWN;
            frontier_size = frontier->count;
        } else {
            if (prev_state == TOP_DOWN)
                top_to_bottom(graph, frontier, bottom_frontier, &clock, v_sets);
            frontier_size = bottom_up_step_fast(graph, bottom_frontier, sol->distances, clock);
            tick_clock(&clock, bottom_frontier, graph->num_nodes);
            if (frontier_size == 0)
                break;
            prev_state = BOTTOM_UP;
        }
        n_nodes_to_explore -= frontier_size;
    }
    delete bottom_frontier;
}
