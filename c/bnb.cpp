#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

#include "node.h"
#include "heap.h"
#include "sparse_mat.h"
#include "solve.h"

#include "data.h"

#include <Highs.h>

void branch(
    Node n, 
    SparseBitMat *A, 
    float *c, 
    NodeHeap *node_heap
) {
    //print_node(&n);
    uint32_t len = n.len;
    int32_t *ties = n.ties;
    float *x = n.x;

    for (uint32_t flip_idx = 0; flip_idx < len; ++flip_idx) {
        if (ties[flip_idx] != -1 || x[flip_idx] == 0.0f) {
            continue;
        }

        Node child = alloc_node(len);
        child.len = len;
        for (uint32_t j = 0; j < flip_idx; ++j) {
            int32_t prev_tied = ties[j];
            if (prev_tied == -1) {
                if (x[j] == 0.0f) {
                    child.ties[j] = -1;
                } else {
                    child.ties[j] = 1;
                }
            } else {
                child.ties[j] = prev_tied;
            }
        }

        child.ties[flip_idx] = 0;

        for (uint32_t j = flip_idx+1; j < len; ++j) {
            int32_t prev_tied = ties[j];
            child.ties[j] = prev_tied;
        }

        solve_node(A, c, &child);
        add_to_heap(node_heap, child);
    }
}

uint32_t branch_and_bound(
    float *c, 
    SparseBitMat *A, 
    uint32_t k,
    Node *best_nodes
) {
    NodeHeap node_heap = init_heap();

    uint32_t best_nodes_len = 0;
    float worst_best_node = 0.0f;
    uint32_t worst_best_idx = 0;

    Node root_node = alloc_node(5);
    for (uint32_t i = 0; i < 5; ++i) {
        root_node.ties[i] = -1;
    }
    root_node.len = 5;

    solve_node(A, c, &root_node);
    add_to_heap(&node_heap, root_node);

    while (node_heap.len != 0) {
        Node node = pop_heap(&node_heap);

        if ((best_nodes_len == k) & (node.cost > worst_best_node)) {
            break;
        }

        bool added = false;
        if (integral(&node)) {
            added = true;

            if (best_nodes_len == k) {
                //printf("%f\n", best_nodes[worst_best_idx].cost);
                //printf("%f\n", node.cost);

                dealloc_node(&best_nodes[worst_best_idx]);
                best_nodes[worst_best_idx] = node;
            } else {
                best_nodes[best_nodes_len] = node;
                ++best_nodes_len;
            }

            worst_best_node = best_nodes[0].cost;
            worst_best_idx = 0;
            for (uint32_t i = 1; i < best_nodes_len; ++i) {
                float cost = best_nodes[i].cost;
                if (cost > worst_best_node) {
                    worst_best_node = cost;
                    worst_best_idx = i;
                }
            }
        }

        branch(node, A, c, &node_heap);

        if (!added) {
            dealloc_node(&node);
        }

    }

    dealloc_heap(&node_heap);

    return best_nodes_len;
}

int main(void) {
    printf("%i\n", sizeof(HighsInt));
    return 0;

//int main_old(void) {
    const float data[15] = {
        0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f
    };

    float c[5] = {-5.4f, -5.0f, -5.0f, -5.0f, -4.0f};
    SparseBitMat A = init_sparse_bit_mat(5, 3, data);
    uint32_t k = 5;

    Node *best_nodes = (Node*)malloc(sizeof *best_nodes * k);

    uint32_t num_solved = branch_and_bound(c, &A, k, best_nodes);
    printf("%d solved\n", num_solved);

    for (uint32_t i = 0; i < num_solved; ++i) {
        float *x = best_nodes[i].x;
        printf("[");
        for (uint32_t x_i = 0; x_i < 4; ++x_i) {
            printf("%.2f ", x[x_i]);
        }
        printf("%.2f] %.2f\n", x[4], best_nodes[i].cost);
    }

    return 0;
}

//int main(void) {
int main_test_heap(void) {
    Node a = alloc_node(5);
    Node b = alloc_node(5);
    Node c = alloc_node(5);
    a.cost = 1.0f;
    b.cost = 2.0f;
    c.cost = 3.0f;

    NodeHeap node_heap = init_heap();
    add_to_heap(&node_heap, c);
    add_to_heap(&node_heap, a);
    add_to_heap(&node_heap, b);

    printf("%d\n", node_heap.len);
    printf("%d\n", node_heap.capacity);

    printf("%f\n", pop_heap(&node_heap).cost);
    printf("%f\n", pop_heap(&node_heap).cost);
    printf("%f\n", pop_heap(&node_heap).cost);

    return 0;
}

//int main(void) {
int main_test_solve(void) {
    const float data[15] = {
        0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f
    };

    float c[5] = {-5.4f, -5.0f, -5.0f, -5.0f, -4.0f};
    SparseBitMat A = init_sparse_bit_mat(5, 3, data);

    int32_t ties[5] = {1, -1, 0, -1, -1};
    Node node = alloc_node(5);
    node.ties = ties;
    node.len = 5;
    solve_node(&A, c, &node);

    float *x = node.x;
    printf("[");
    for (uint32_t x_i = 0; x_i < 4; ++x_i) {
        printf("%.2f ", x[x_i]);
    }
    printf("%.2f] %.2f\n", x[4], node.cost);

    return 0;
}

//int main(void) {
int main_test_mul(void) {
    const float data[15] = {
        0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f
    };

    SparseBitMat A = init_sparse_bit_mat(5, 3, data);

    float x[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float result[3];

    mat_vec_mul(&A, x, result);

    printf("%f\n", result[0]);
    printf("%f\n", result[1]);
    printf("%f\n", result[2]);

    return 0;
}

int main_test_branch(void) {
//int main(void) {
    const float data[15] = {
        0.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        0.0f, 1.0f, 1.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 0.0f, 1.0f, 0.0f
    };

    float c[5] = {-5.4f, -5.0f, -5.0f, -5.0f, -4.0f};
    SparseBitMat A = init_sparse_bit_mat(5, 3, data);

    int32_t ties[5] = {-1, -1, -1, -1, -1};
    Node node = alloc_node(5);
    node.ties = ties;
    node.len = 5;
    solve_node(&A, c, &node);

    NodeHeap node_heap = init_heap();

    branch(node, &A, c, &node_heap);

    for (uint32_t i = 0; i < node_heap.len; ++i) {
        print_node(&node_heap.data[i]);
    }

    return 0;
}
