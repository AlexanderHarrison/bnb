#pragma once

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

typedef struct {
    float cost;
    uint32_t len;
    float *x;
    int32_t *ties;
} Node;

void dealloc_node(Node *node) {
    free(node->x);
}

void print_node(const Node *node) {
    float *x = node->x;
    int32_t *ties = node->ties;
    uint32_t last = node->len - 1;

    printf("[");
    for (uint32_t x_i = 0; x_i < last; ++x_i) {
        printf("%i ", ties[x_i]);
    }
    printf("%i] [", ties[last]);

    for (uint32_t x_i = 0; x_i < last; ++x_i) {
        printf("%.2f ", x[x_i]);
    }
    printf("%.2f] %.2f\n", x[last], node->cost);
}

Node alloc_node(uint32_t len) {
    Node node;
    float *x = (float*)malloc(sizeof *x * len * 2);
    node.x = x;
    node.ties = (int32_t *)(x + len);
    return node;
}

bool integral(const Node *node) {
    float *x = node->x;
    uint32_t len = node->len;

    for (uint32_t i = 0; i < len; ++i) {
        float v = x[i];
        if ((v != 1.0f) & (v != 0.0f)) {
            return false;
        }
    }
    return true;
}

void reorder(Node *node, const uint32_t *order) {
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
