#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

typedef struct {
    uint32_t len;
    float cost;
    float *x;
    int32_t *ties;
} Node;

typedef struct {
    uint16_t col;
    uint16_t row;
} MatBit;

typedef struct {
    MatBit *bits;
    uint32_t bitcount;
} SparseMat;

typedef struct {
    int temp;
} NodeHeap;

inline void dealloc_node(Node *node) {
    free(node->x);
}

inline Node alloc_node(uint32_t len) {
    Node node;
    float *x = malloc(sizeof *x * len * 2);
    node.x = x;
    node.ties = (uint32_t *)(x + len);
    return node;
}

inline bool integral(Node node) {
    for (uint32_t i = 0; i < node.len; ++i) {
        float v = node.x[i];
        if ((v != 1.0f) & (v != 0.0f)) {
            return false;
        }
    }
    return true;
}

inline bool lt(Node *a, Node *b) {
    return a->cost < b->cost;
}

void reorder(Node *node, uint32_t *order) {
    uint32_t len = node->len;
    Node node_new = alloc_node(len);

    float *x = node->x;
    int32_t *ties = node->ties;

    for (uint32_t i = 0; i < len; ++i) {
        node_new.x[i] = x[order[i]];
    }

    for (uint32_t i = 0; i < len; ++i) {
        node_new.ties[i] = ties[order[i]];
    }

    dealloc_node(node);
    *node = node_new;
}

Node *branch(
    Node n, 
    SparseMat A, 
    float *c, 
    uint32_t k, 
    uint32_t *scratch,
    Heap *node_heap,
) {
    uint32_t len = n.len;
    int32_t *ties = n.ties;
    float *x = n.x;

    uint32_t num_untied_nonzero = 0;
    uint32_t *untied_nonzero_idx = scratch;
    for (uint32_t i = 0; i < n.len; ++i) {
        if (ties[i] == -1 && x[i] > 0.0f) {
            untied_nonzero_idx[num_untied_nonzero] = i;
            ++num_untied_nonzero;
        }
    }

    for (uint32_t i = 0; i < num_untied_nonzero; ++i) {
        Node child = alloc_node(len);
        uint32_t flip_idx = untied_nonzero_idx[i];

        for (uint32_t j = 0; j < flip_idx; ++j) {
            uint32_t prev_tied = ties[j];
            if (prev_tied == -1) {
                if (x[j] == 0.0f) {
                    child_ties[j] = -1;
                } else {
                    child_ties[j] = 1.0f;
                }
            } else {
                child.ties[j] = prev_tied;
            }
        }

        child.ties[flip_idx] = 0;

        for (uint32_t j = flip_idx+1; j < len; ++j) {
            uint32_t prev_tied = ties[j];
            child.ties[j] = prev_tied;
        }

        // TODO
    }

    return NULL;
}

inline void add_to_heap(NodeHeap *heap, Node node) {
    // TODO
}

inline void solve_node(Node *node) {
    // TODO
}
