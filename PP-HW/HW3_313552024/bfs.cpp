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

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    #pragma omp parallel
    {
        // Each thread maintains its own local frontier
        vertex_set local_frontier;
        vertex_set_init(&local_frontier, g->num_nodes);

        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < frontier->count; i++)
        {
            int node = frontier->vertices[i];
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->outgoing_starts[node + 1];

            // Check all neighbors
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];

                // Use atomic operation for thread safety
                if (distances[outgoing] == NOT_VISITED_MARKER) {
                    if (__sync_bool_compare_and_swap(&distances[outgoing], 
                        NOT_VISITED_MARKER, distances[node] + 1))
                    {
                        // 只有成功更新距離的執行緒才加入 frontier
                        int index = local_frontier.count++;
                        local_frontier.vertices[index] = outgoing;
                    }
                }
            }
        }
        // Merge local results
        if (local_frontier.count > 0) {
            #pragma omp critical
            {
                memcpy(new_frontier->vertices + new_frontier->count,
                       local_frontier.vertices,
                       local_frontier.count * sizeof(int));
                new_frontier->count += local_frontier.count;
            }
        }
        free(local_frontier.vertices);
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // Initialize distances
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // Setup root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

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
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

// Bottom-up BFS step implementation
void bottom_up_step(
    Graph g,
    int *frontier,
    int frontier_count,
    int *new_frontier,
    int *new_frontier_count,
    int *distances,
    int level)
{
    #pragma omp parallel
    {
        int local_count = 0;
        int *local_frontier = (int *)malloc(sizeof(int) * g->num_nodes);

        #pragma omp for schedule(dynamic, 1024)
        for (int i = 0; i < g->num_nodes; i++)
        {
            if (distances[i] == NOT_VISITED_MARKER)
            {
                int start_edge = g->incoming_starts[i];
                int end_edge = (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];
                for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
                {
                    int incoming = g->incoming_edges[neighbor];
                    if (distances[incoming] == level)
                    {
                        distances[i] = level + 1;
                        local_frontier[local_count++] = i;
                        break;
                    }
                }
            }
        }
        if (local_count > 0) {
            int offset;
            #pragma omp critical
            {
                offset = *new_frontier_count;
                *new_frontier_count += local_count;
            }
            memcpy(new_frontier + offset, local_frontier, local_count * sizeof(int));
        }
        free(local_frontier);
    }
}

// Bottom-up BFS implementation
void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    // 初始化距離陣列
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    sol->distances[ROOT_NODE_ID] = 0;
    
    int *frontier = (int *)malloc(sizeof(int) * graph->num_nodes);
    int *new_frontier = (int *)malloc(sizeof(int) * graph->num_nodes);
    
    frontier[0] = ROOT_NODE_ID;
    int frontier_count = 1;
    int level = 0;
    while (frontier_count > 0)
    {
        int new_frontier_count = 0;

        bottom_up_step(graph, frontier, frontier_count, new_frontier, 
                      &new_frontier_count, sol->distances, level);

        // Swap frontiers
        int *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        frontier_count = new_frontier_count;
        level++;
    }

    free(frontier);
    free(new_frontier);
}

// Hybrid BFS implementation combining top-down and bottom-up approaches
void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1, list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // Initialize distances
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int level = 0;
    bool is_bottom_up = false;

    // Thresholds for switching between approaches
    float alpha = 14.0;
    float beta = 24.0;

    // Adjust thresholds based on graph size
    if (graph->num_nodes > 10000000) {  // random_500m.graph
        alpha = 10.0;
        beta = 18.0;
    } else if (graph->num_nodes > 1000000) {  // com-orkut_117m.graph
        alpha = 22.0;
        beta = 40.0;
    } else if (graph->num_nodes > 100000) {  // soc-livejournal1_68m.graph
        alpha = 18.0;
        beta = 32.0;
    }

    while (frontier->count != 0)
    {
        vertex_set_clear(new_frontier);

        // Calculate edge density
        float edges_to_check = 0;

        if (frontier->count < graph->num_nodes/15) {
            edges_to_check = frontier->count;
            // Adjust sampling strategy based on graph size
            int sample_size;
            if (graph->num_nodes > 10000000) {
                sample_size = (frontier->count > 1000) ? 1000 : frontier->count;
            } else if (graph->num_nodes > 1000000) {
                sample_size = (frontier->count > 200) ? 200 : frontier->count;
            } else {
                sample_size = (frontier->count > 400) ? 400 : frontier->count;
            }
            
            // Sample edge count
            for (int i = 0; i < sample_size; i++) {
                int node = frontier->vertices[i];
                edges_to_check += graph->outgoing_starts[node + 1] - 
                                graph->outgoing_starts[node];
            }
            
            if (sample_size < frontier->count) {
                edges_to_check = edges_to_check * frontier->count / sample_size;
            }

            // Special handling for different graph types
            if (graph->num_nodes > 1000000 && graph->num_nodes <= 10000000) {
                float level_factor = (level < 2) ? 2.5 : 
                                   (level < 4) ? 1.8 : 1.0;
                if (!is_bottom_up && 
                    edges_to_check > graph->num_nodes / (alpha * level_factor) &&
                    frontier->count > graph->num_nodes/40) {
                    is_bottom_up = true;
                } else if (is_bottom_up && 
                         (edges_to_check < graph->num_nodes / (beta * level_factor) ||
                          frontier->count < graph->num_nodes/35)) {
                    is_bottom_up = false;
                }
            } else {
                float level_factor = (level < 3) ? 1.8 : 1.0;
                if (!is_bottom_up && edges_to_check > graph->num_nodes / (alpha * level_factor)) {
                    is_bottom_up = true;
                } else if (is_bottom_up && edges_to_check < graph->num_nodes / (beta * level_factor)) {
                    is_bottom_up = false;
                }
            }
        }

        // Force top-down for small frontiers
        if (is_bottom_up) {
            if (graph->num_nodes > 1000000 && graph->num_nodes <= 10000000) {
                if (frontier->count < graph->num_nodes/25) {
                    is_bottom_up = false;
                }
            } else if (graph->num_nodes > 100000 && graph->num_nodes <= 1000000) {
                if (frontier->count < graph->num_nodes/40) {
                    is_bottom_up = false;
                }
            } else if (frontier->count < graph->num_nodes/100) {
                is_bottom_up = false;
            }
        }

        // Choose and execute appropriate BFS approach
        if (is_bottom_up) {
            // Bottom-up
            int *temp_frontier = (int *)malloc(sizeof(int) * graph->num_nodes);
            int temp_count = 0;
            
            bottom_up_step(graph, frontier->vertices, frontier->count,
                          temp_frontier, &temp_count, sol->distances, level);

            new_frontier->count = temp_count;
            memcpy(new_frontier->vertices, temp_frontier, 
                   temp_count * sizeof(int));
            free(temp_frontier);
        } else {
            // Top-down
            top_down_step(graph, frontier, new_frontier, sol->distances);
        }
        level++;
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    free(list1.vertices);
    free(list2.vertices);
}
